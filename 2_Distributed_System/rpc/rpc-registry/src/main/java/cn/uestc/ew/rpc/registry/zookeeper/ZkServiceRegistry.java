package cn.uestc.ew.rpc.registry.zookeeper;

import cn.uestc.ew.rpc.registry.ServiceRegistry;
import org.I0Itec.zkclient.ZkClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ZooKeeper 服务注册类，为服务端提供服务注册
 */
public class ZkServiceRegistry implements ServiceRegistry {

    private static final Logger LOGGER = LoggerFactory.getLogger(ZkServiceRegistry.class);

    /**
     * Zookeeper Client
     */
    private final ZkClient zkClient;

    public ZkServiceRegistry(String zkAddress) {
        // 创建 ZooKeeper 客户端
        zkClient = new ZkClient(zkAddress, Constants.SESSION_TIMEOUT, Constants.CONNECTION_TIMEOUT);
        LOGGER.debug("connect zookeeper");
    }

    @Override
    public void register(String serviceName, String serviceAddress) {
        // 1. 创建持久化的 registry 节点
        String registryPath = Constants.REGISTRY_ROOT_PATH;
        if (!zkClient.exists(registryPath)) {
            zkClient.createPersistent(registryPath);
            LOGGER.debug("Create registry node: {}", registryPath);
        }
        // 2. 创建持久化的 service 节点
        String servicePath = registryPath + "/" + serviceName;
        if (!zkClient.exists(servicePath)) {
            zkClient.createPersistent(servicePath);
            LOGGER.debug("Create service node: {}", servicePath);
        }
        // 3. 创建临时的 address 节点
        String addressPath = servicePath + "/address-";
        String addressNode = zkClient.createEphemeralSequential(addressPath, serviceAddress);
        LOGGER.debug("Create address node: {}", addressNode);
    }
}