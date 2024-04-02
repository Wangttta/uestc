package cn.uestc.ew.rpc.registry;

/**
 * 服务发现接口，为 RPC 客户端提供服务发现功能
 */
public interface ServiceDiscovery {

    /**
     * 服务发现接口
     *
     * @param serviceName 查找的服务名称
     * @return 对应的服务地址
     */
    String discover(String serviceName);
}