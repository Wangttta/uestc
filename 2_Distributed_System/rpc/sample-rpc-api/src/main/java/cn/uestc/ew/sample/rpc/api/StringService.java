package cn.uestc.ew.sample.rpc.api;

/**
 * 样例：字符串服务 API
 */
public interface StringService {

    /**
     * 将给定字符串转换为大写
     */
    String toUppercase(String text);
}