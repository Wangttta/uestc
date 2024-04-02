package cn.uestc.ew.rpc.common.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class RpcConfig {

    /**
     * 连接超时时间，单位毫秒
     */
    private int timeout = 5000;

    /**
     * 超时重试策略
     *
     * <ul>
     *   <li>0：不使用超时重试，即 At-most-once </li>
     * </ul>
     */
    private int retryPolicy = 1;

    /**
     * 超时重试次数，设置为 -1 表示无限次重试
     */
    private int retryTimes = 10;

    /**
     * 是否关闭了重试策略
     * @return 当没有配置重试时，返回 {@code true}，否则返回 {@code false}
     */
    public boolean noRetry() {
        return retryPolicy == 0;
    }

    /**
     * 返回第 {@code i} 次请求是否需要发送，本次请求如果超出了最大重试次数，
     * 则返回 {@code false} 表示取消发送
     *
     * @param i 当前是第几次重试，一般是一个非负整数且小于等于 {@link RpcConfig#retryTimes}。
     *          当超时重试次数配置为 -1 时忽略当前参数
     * @return 本次重试请求是否需要发送，返回 {@code true or false}
     */
    public boolean needRetry(int i) {
        if (noRetry()) return false;
        else {
            if (retryTimes == -1) return true;
            else return i <= retryTimes;
        }
    }
}