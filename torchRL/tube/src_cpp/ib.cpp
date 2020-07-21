
#include "rdma.h"

#include <infiniband/verbs.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <iostream>

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

struct Port {
  int num = 0;
  ibv_port_attr attr;
  uint32_t lid() {
    return attr.lid;
  }
};

struct Context {
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

struct ProtectionDomain {
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

struct MemoryRegion {
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

struct CompletionQueue {
  ibv_cq* cq = nullptr;
  CompletionQueue(const Context& ctx) {
    cq = ibv_create_cq(ctx.context, 64, nullptr, nullptr, 0);
    if (!cq) {
      throw Error("Failed to create completion queue");
    }
  }
  ~CompletionQueue() {
    if (cq) {
      ibv_destroy_cq(cq);
    }
  }

  void wait() {
    ibv_wc wc;
    int n = -1;
    while ((n = ibv_poll_cq(cq, 1, &wc)) == 0)
      ;
    if (n < 0) {
      throw Error("Failed to poll the completition queue");
    }
    if (wc.status != IBV_WC_SUCCESS) {
      throw Error("Failed with status: " +
                  std::string(ibv_wc_status_str(wc.status)));
    }
  }
};

struct QueuePair {
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
        qp,
        &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (err) {
      throw Error("Failed to move queue pair to init state; error " +
                  std::to_string(err));
    }
  }

  void rtr(const Port& port, uint16_t remoteLid, uint32_t remoteQPNum) {
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
    int err = ibv_modify_qp(
        qp,
        &attr,
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
    attr.retry_cnt = 4;
    attr.rnr_retry = 4;
    attr.max_rd_atomic = 1;
    int err = ibv_modify_qp(qp,
                            &attr,
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
};

struct ibHost : Host {
  ib::Context* context = nullptr;
  ib::ProtectionDomain* pd = nullptr;
  std::optional<ib::Port> port;
  std::optional<ib::CompletionQueue> cq;
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
  }
  virtual Endpoint init() override {
    cq.emplace(*context);
    qp.emplace(*pd, *cq);
    qp->init(*port, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    inRts = false;
    return {port->lid(), qp->num()};
  }
  virtual void connect(Endpoint ep) override {
    qp->rtr(*port, ep.lid, ep.qpnum);
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
    auto& buf = (ibBuffer&)localBuffer;

    qp->read(*buf.mr, localAddress, remoteKey, remoteAddress, size);
  }
  virtual void wait() override {
    cq->wait();
  }
};

struct ibContext : Context {
  ib::DeviceList devlist;
  std::optional<ib::Context> context;
  std::optional<ib::ProtectionDomain> pd;
  ibContext() {
    if (devlist.empty()) {
      throw ib::Error("No infiniband devices found");
    }
    context.emplace(devlist[0]);
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
};

std::unique_ptr<Context> create() {
  return std::make_unique<ibContext>();
}

}  // namespace rdma
