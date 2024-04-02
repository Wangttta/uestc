package cn.uestc.ew.rpc.registry;

/**
 * 服务注册接口，为 RPC 服务器端提供服务注册功能
 */
public interface ServiceRegistry {

    /**
     * 服务注册接口
     *
     * @param serviceName       待注册服务名称
     * @param serviceAddress    待注册服务地址
     */
    void register(String serviceName, String serviceAddress);
}