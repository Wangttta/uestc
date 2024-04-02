package cn.uestc.ew.rpc.registry.zookeeper;

public interface Constants {

    /**
     * RPC 服务在 Zookeeper 的默认注册根路径
     */
    String REGISTRY_ROOT_PATH = "/rpc";

    /**
     * ZooKeeper 在客户端连接断开 SESSION_TIMEOUT ms 后未重新连接， 则将 Session 标记位过期
     */
    int SESSION_TIMEOUT = 500000;

    /**
     * ZooKeeper 客户端发起连接 CONNECTION_TIMEOUT ms 后未收到响应，标记为连接失败
     */
    int CONNECTION_TIMEOUT = 100000;
}