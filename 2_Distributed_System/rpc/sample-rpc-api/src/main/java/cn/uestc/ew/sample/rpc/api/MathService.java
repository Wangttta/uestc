package cn.uestc.ew.sample.rpc.api;

/**
 * 样例：数学计算服务 API
 */
public interface MathService {

    /**
     * 整数加法运算
     *
     * @param a 第一操作数
     * @param b 第二操作数
     * @return 返回表达式 (a + b) 的结果
     */
    int sum(int a, int b);

    /**
     * 浮点数加法运算
     *
     * @param a 第一操作数
     * @param b 第二操作数
     * @return 返回表达式 (a + b) 的结果
     */
    float sum(float a, float b);
}