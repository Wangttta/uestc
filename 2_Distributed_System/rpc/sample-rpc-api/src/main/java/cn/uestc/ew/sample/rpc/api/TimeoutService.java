package cn.uestc.ew.sample.rpc.api;

/**
 * 样例：超时服务 API，测试 At-least-once 语义
 */
public interface TimeoutService {

    /**
     * 此接口初次调用会超时（10000ms），后续调用恢复正常
     */
    String timeoutAtFirstTime();
}