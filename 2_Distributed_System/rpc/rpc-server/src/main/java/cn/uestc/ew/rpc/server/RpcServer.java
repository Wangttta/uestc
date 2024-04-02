package cn.uestc.ew.rpc.server;

public interface RpcServer {

    /**
     * RPC 服务端启动方法，可基于不同的网络通信技术实现如 Socket BIO、Socket NIO、Netty 等
     */
    void startup();
}