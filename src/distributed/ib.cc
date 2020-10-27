
#include "rdma.h"

#include <infiniband/verbs.h>

#include <deque>
#include <list>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ib {

struct Device {
  std::string name;
  ibv_device* info;
};

struct DeviceList {

  std::vector<Device> list;
  ibv_device** rawlist = nullptr;

  DeviceList() {
    int num = 0;
    rawlist = ibv_get_device_list(&num);
    for (int i = 0; i < num; ++i) {
      Device d;
      d.name = rawlist[i]->name;
      d.info = rawlist[i];
      list.push_back(d);
    }
  }
  ~DeviceList() {
    if (rawlist) {
      ibv_free_device_list(rawlist);
    }
  }

  size_t size() const {
    return list.size();
  }

  auto begin() const {
    return list.begin();
  }
  auto end() const {
    return list.end();
  }

  bool empty() const {
    return list.empty();
  }

  decltype(auto) operator[](size_t index) const {
    return list[index];
  }
};

class Error : public rdma::Error {
 public:
  using rdma::Error::Error;
};

std::string gidstr(std::array<std::byte, 16> gid) {
  std::string s;
  for (auto& v : gid) {
    s += "0123456789abcdef"[unsigned(v) >> 4];
    s += "0123456789abcdef"[unsigned(v) & 0xf];
  }
  return s;
}

struct Port {
  ibv_context* context = nullptr;
  int num = 0;
  ibv_port_attr attr;
  uint32_t lid() {
    return attr.lid;
  }
  std::array<std::byte, 16> gid() {
    std::array<std::byte, 16> gid;
    static_assert(sizeof(gid) == sizeof(ibv_gid));
    if (ibv_query_gid(context, num, 0, (ibv_gid*)&gid)) {
      throw Error("ibv_query_gid failed");
    }
    return gid;
  }
};

struct NoMove {
  NoMove() = default;
  NoMove(const NoMove&) = delete;
  NoMove(NoMove&&) = delete;
  NoMove& operator=(const NoMove&) = delete;
  NoMove& operator=(NoMove&&) = delete;
};

struct Context : NoMove {
  ibv_context* context = nullptr;
  ibv_device_attr devattr;
  std::vector<Port> ports;
  Context(const Device& dev) {
    context = ibv_open_device(dev.info);
    if (!context) {
      throw Error("Failed to open device " + dev.name);
    }
    memset(&devattr, 0, sizeof(devattr));
    ibv_query_device(context, &devattr);

    for (int i = 1; i <= devattr.phys_port_cnt; ++i) {
      ports.emplace_back();
      ports.back().context = context;
      ports.back().num = i;
      auto& attr = ports.back().attr;
      memset(&attr, 0, sizeof(attr));
      ibv_query_port(context, i, &attr);
    }
  }
  ~Context() {
    if (context) {
      ibv_close_device(context);
      context = nullptr;
    }
  }
};

struct ProtectionDomain : NoMove {
  ibv_pd* pd = nullptr;
  ProtectionDomain(Context& ctx) {
    pd = ibv_alloc_pd(ctx.context);
    if (!pd) {
      throw Error("Failed to allocate protection domain");
    }
  }
  ~ProtectionDomain() {
    if (pd) {
      ibv_dealloc_pd(pd);
      pd = nullptr;
    }
  }
};

struct MemoryRegion : NoMove {
  ibv_mr* mr = nullptr;
  MemoryRegion(const ProtectionDomain& pd,
               void* address,
               size_t size,
               int access) {
    mr = ibv_reg_mr(pd.pd, address, size, access);
    if (!mr) {
      throw Error("Failed to register memory region");
    }
  }
  ~MemoryRegion() {
    if (mr) {
      ibv_dereg_mr(mr);
      mr = nullptr;
    }
  }
  auto lkey() {
    return mr->lkey;
  }
  auto rkey() {
    return mr->rkey;
  }
};

struct CompletionQueue : NoMove {
  ibv_cq* cq = nullptr;
  CompletionQueue(const Context& ctx, int size) {
    cq = ibv_create_cq(ctx.context, size, nullptr, nullptr, 0);
    if (!cq) {
      throw Error("Failed to create completion queue");
    }
  }
  ~CompletionQueue() {
    if (cq) {
      int e = ibv_destroy_cq(cq);
      if (e) {
        throw std::runtime_error("ibv_destroy_cq failed with error " +
                                 std::to_string(e));
      }
      cq = nullptr;
    }
  }

  void wait() {
    ibv_wc wc;
    auto start = std::chrono::steady_clock::now();
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (std::chrono::steady_clock::now() - start >=
          std::chrono::seconds(10)) {
        throw Error("wait timed out");
      }
      int r = ibv_poll_cq(cq, 1, &wc);
      if (r > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          throw Error(ibv_wc_status_str(wc.status));
        }
        return;
      } else if (r < 0) {
        throw Error("Failed to poll the completion queue");
      }
    }
  }
};

struct QueuePair : NoMove {
  ibv_qp* qp = nullptr;
  QueuePair(const ProtectionDomain& pd, const CompletionQueue& cq) {
    ibv_qp_init_attr init;
    memset(&init, 0, sizeof(init));
    init.send_cq = cq.cq;
    init.recv_cq = cq.cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 2;
    init.cap.max_recv_wr = 2;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;

    qp = ibv_create_qp(pd.pd, &init);
    if (!qp) {
      throw Error("Failed to create queue pair");
    }
  }
  ~QueuePair() {
    if (qp) {
      ibv_destroy_qp(qp);
    }
  }

  uint32_t num() {
    return qp->qp_num;
  }

  void init(const Port& port, int accessFlags) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port.num;
    attr.qp_access_flags = accessFlags;
    int err = ibv_modify_qp(
        qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (err) {
      throw Error("Failed to move queue pair to init state; error " +
                  std::to_string(err));
    }
  }

  void rtr(const Port& port,
           uint16_t remoteLid,
           uint32_t remoteQPNum,
           const std::array<std::byte, 16>& gid) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port.attr.active_mtu;
    attr.dest_qp_num = remoteQPNum;
    attr.rq_psn = 4242;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remoteLid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port.num;

    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 4;
    attr.ah_attr.grh.dgid = (ibv_gid&)gid;
    attr.ah_attr.grh.sgid_index = 0;
    int err = ibv_modify_qp(
        qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (err) {
      throw Error("Failed to move queue pair to rtr state; error " +
                  std::to_string(err));
    }
  }

  void rts() {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4242;
    attr.timeout = 17;  // 0.5s
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    int err = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                IBV_QP_MAX_QP_RD_ATOMIC);
    if (err) {
      throw Error("Failed to move queue pair to rts state; error " +
                  std::to_string(err));
    }
  }

  void read(MemoryRegion& dstmr,
            void* dstbuf,
            uint32_t rkey,
            uintptr_t remoteAddress,
            size_t length) {

    ibv_sge sg;
    ibv_send_wr wr;
    ibv_send_wr* bad_wr;

    memset(&sg, 0, sizeof(sg));
    sg.addr = (uintptr_t)dstbuf;
    sg.length = length;
    sg.lkey = dstmr.lkey();

    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remoteAddress;
    wr.wr.rdma.rkey = rkey;

    ibv_qp_attr qattr;
    ibv_qp_init_attr qiattr;

    if (ibv_query_qp(qp, &qattr, IBV_QP_STATE, &qiattr)) {
      throw Error("Failed to query qp");
    }

    int err = ibv_post_send(qp, &wr, &bad_wr);
    if (err) {
      throw Error("RDMA read failed; error " + std::to_string(err));
    }
  }
};

}  // namespace ib

namespace rdma {

struct ibBuffer : Buffer {
  std::optional<ib::MemoryRegion> mr;
  virtual ~ibBuffer() override {
  }
  virtual uint32_t key() override {
    return mr->rkey();
  }
  virtual uint32_t keyFor(Endpoint ep) override {
    return mr->rkey();
  }
};

struct ibCompletionQueue : CompletionQueue {
  ib::CompletionQueue cq;
  virtual ~ibCompletionQueue() {
  }
  ibCompletionQueue(ib::Context& ctx, int size)
      : cq(ctx, size) {
  }
  virtual void wait() override {
    cq.wait();
  }
};

struct ibMultiBuffer : Buffer {
  std::deque<ib::MemoryRegion> mrs;
  std::vector<uint32_t> lids;
  virtual ~ibMultiBuffer() override {
  }
  virtual uint32_t key() override {
    std::abort();
  }
  ib::MemoryRegion& mrFor(Endpoint ep) {
    for (size_t i = 0; i != lids.size(); ++i) {
      if (lids[i] == ep.lid) {
        return mrs.at(i);
      }
    }
    throw Error("Endpoint not found for multibuffer");
  }
  virtual uint32_t keyFor(Endpoint ep) override {
    return mrFor(ep).rkey();
  }
};

struct ibHost : Host {
  ib::Context* context = nullptr;
  ib::ProtectionDomain* pd = nullptr;
  std::optional<ib::Port> port;
  ib::CompletionQueue* cq = nullptr;
  std::optional<ib::QueuePair> qp;
  bool inRts = false;
  virtual ~ibHost() override {
  }
  ibHost(ib::Context* context, ib::ProtectionDomain* pd)
      : context(context)
      , pd(pd) {
    if (context->ports.empty()) {
      throw ib::Error("Infiniband device has no ports!");
    }
    port = context->ports.at(0);
    // printf("Host using %d:%d\n", port->lid(), port->num);
  }
  virtual Endpoint init(CompletionQueue& cqa) override {
    cq = &((ibCompletionQueue&)cqa).cq;
    qp.emplace(*pd, *cq);
    qp->init(*port, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    inRts = false;
    // printf("QP %d:%d\n", port->lid(), qp->num());
    return {port->lid(), qp->num(), port->gid()};
  }
  virtual void connect(Endpoint ep) override {
    qp->rtr(*port, ep.lid, ep.qpnum, ep.gid);
    inRts = false;
  }
  virtual void read(Buffer& localBuffer,
                    void* localAddress,
                    uint32_t remoteKey,
                    uintptr_t remoteAddress,
                    size_t size) override {
    if (!inRts) {
      qp->rts();
      inRts = true;
    }
    if (auto buf = dynamic_cast<ibBuffer*>(&localBuffer)) {
      qp->read(*buf->mr, localAddress, remoteKey, remoteAddress, size);
    } else if (auto buf = dynamic_cast<ibMultiBuffer*>(&localBuffer)) {
      qp->read(buf->mrFor(Endpoint{port->lid(), qp->num(), port->gid()}),
               localAddress, remoteKey, remoteAddress, size);
    }
  }
  virtual void wait() override {
    cq->wait();
  }
};

struct ibContext : Context {
  ib::Device device;
  std::optional<ib::Context> context;
  std::optional<ib::ProtectionDomain> pd;
  ibContext(ib::Device device)
      : device(device) {
    context.emplace(device);
    pd.emplace(*context);
  }
  virtual ~ibContext() override {
  }
  virtual std::unique_ptr<Host> createHost() override {
    return std::make_unique<ibHost>(&*context, &*pd);
  }
  virtual std::unique_ptr<Buffer> createBuffer(void* address,
                                               size_t size) override {
    auto r = std::make_unique<ibBuffer>();
    r->mr.emplace(
        *pd, address, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    return r;
  }
  virtual std::unique_ptr<CompletionQueue> createCQ(int size) override {
    return std::make_unique<ibCompletionQueue>(*context, size);
  }
};

struct ibMultiCompletionQueue : CompletionQueue {
  std::vector<std::shared_ptr<ibCompletionQueue>> cqs;
  virtual ~ibMultiCompletionQueue() override {
  }
  virtual void wait() override {
    std::abort();
  }
};

struct ibMultiHost : ibHost {
  size_t index;
  ibMultiHost(size_t index, ib::Context* context, ib::ProtectionDomain* pd)
      : ibHost(context, pd)
      , index(index) {
  }
  virtual ~ibMultiHost() override {
  }
  virtual Endpoint init(CompletionQueue& cqa) override {
    auto cq = (ibMultiCompletionQueue&)cqa;
    return ibHost::init(*cq.cqs.at(index));
  }
};

struct ibMultiContext : Context {
  ib::DeviceList devlist;
  std::deque<ibContext> contexts;
  std::minstd_rand rng;
  ibMultiContext() {
    if (devlist.empty()) {
      throw ib::Error("No infiniband devices found");
    }
    rng.seed(std::random_device{}());
    for (auto& v : devlist) {
      contexts.emplace_back(v);
    }
  }
  virtual ~ibMultiContext() override {
  }
  virtual std::unique_ptr<Host> createHost() override {
    size_t index =
        std::uniform_int_distribution<size_t>(0, contexts.size() - 1)(rng);
    auto& ctx = contexts[index];
    return std::make_unique<ibMultiHost>(index, &*ctx.context, &*ctx.pd);
  }
  virtual std::unique_ptr<Buffer> createBuffer(void* address,
                                               size_t size) override {
    auto r = std::make_unique<ibMultiBuffer>();
    for (auto& ctx : contexts) {
      r->mrs.emplace_back(*ctx.pd, address, size,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
      r->lids.push_back(ctx.context->ports.at(0).lid());
    }
    return r;
  }
  virtual std::unique_ptr<CompletionQueue> createCQ(int size) override {
    auto r = std::make_unique<ibMultiCompletionQueue>();
    for (auto& ctx : contexts) {
      r->cqs.push_back(std::make_unique<ibCompletionQueue>(*ctx.context, size));
    }
    return r;
  }
};

std::unique_ptr<Context> create() {
  return std::make_unique<ibMultiContext>();
}

}  // namespace rdma
