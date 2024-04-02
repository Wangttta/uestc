package cn.uestc.ew.rpc.common.util.codec;

import lombok.experimental.UtilityClass;

/**
 * 自定义编码器
 */
@UtilityClass
public class RpcEncoder {

    /**
     * 整型数字转字节数组，按照小端法保存
     *
     * <pre>以十进制数 120 为例，下面展示了它的转换方法：
     *     120 (10) = 0b 00000000 00000000 00000000 01111000
     *              => byte[4]{0b01111000, 0b00000000, 0b00000000, 0b00000000}
     * </pre>
     *
     * @param n 需要转换字节数组的整型数字
     * @return 转换后的字节数组
     */
    public static byte[] toBytes(int n) {
        byte[] b = new byte[4];
        b[0] = (byte) (n & 0xff);
        b[1] = (byte) (n >> 8 & 0xff);
        b[2] = (byte) (n >> 16 & 0xff);
        b[3] = (byte) (n >> 24 & 0xff);
        return b;
    }
}