package cn.uestc.ew.rpc.registry.zookeeper;

import cn.uestc.ew.rpc.registry.ServiceDiscovery;
import org.I0Itec.zkclient.ZkClient;
import org.apache.commons.collections4.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * ZooKeeper 服务发现类，为客户端提供服务发现
 */
public class ZkServiceDiscovery implements ServiceDiscovery {

    private static final Logger LOGGER = LoggerFactory.getLogger(ZkServiceDiscovery.class);

    /**
     * ZooKeeper 服务器地址，通过配置传入
     */
    private final String zkAddress;

    public ZkServiceDiscovery(String zkAddress) {
        this.zkAddress = zkAddress;
    }

    @Override
    public String discover(String name) {
        // 1. 创建 ZooKeeper 客户端
        ZkClient zkClient = new ZkClient(zkAddress, Constants.SESSION_TIMEOUT, Constants.CONNECTION_TIMEOUT);
        LOGGER.debug("Connect zookeeper start: address={}", zkAddress);
        try {
            // 2. 获取 Service 节点
            String servicePath = Constants.REGISTRY_ROOT_PATH + "/" + name;
            if (!zkClient.exists(servicePath)) {
                throw new RuntimeException(String.format("Can not find any service node on path: %s", servicePath));
            }
            List<String> addressList = zkClient.getChildren(servicePath);
            if (CollectionUtils.isEmpty(addressList)) {
                throw new RuntimeException(String.format("Can not find any address node on path: %s", servicePath));
            }
            // 3. 获取 Address 节点
            String address;
            int size = addressList.size();
            if (size == 1) {
                // 该服务只被注册了一次
                address = addressList.get(0);
                LOGGER.debug("Find single address node: {}", address);
            } else {
                // 该服务被注册了多次，随机获其中
                // TODO: 2022/10/3 Load balance
                address = addressList.get(ThreadLocalRandom.current().nextInt(size));
                LOGGER.debug("Find random service node: {}", address);
            }
            // 4. 获取 address 节点的值
            String addressPath = servicePath + "/" + address;
            return zkClient.readData(addressPath);
        } finally {
            zkClient.close();
        }
    }
}