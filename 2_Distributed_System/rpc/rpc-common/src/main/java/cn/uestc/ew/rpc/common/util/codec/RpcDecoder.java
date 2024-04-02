package cn.uestc.ew.rpc.common.util.codec;

import cn.uestc.ew.rpc.common.exception.Asserts;
import lombok.experimental.UtilityClass;

/**
 * 自定义解码器
 */
@UtilityClass
public class RpcDecoder {

    /**
     * 字节数组转整型数字，按照小端法转换
     *
     * <pre>以十进制数 120 为例，下面展示了它的转换方法：
     *      byte[4]{0b01111000, 0b00000000, 0b00000000, 00000b0000}
     *      = 0b 00000000 00000000 00000000 01111000
     *      = 120 (10)
     * </pre>
     *
     * @param bytes 需要转换整型数字的字节数组
     * @return 转换后的整型数字
     */
    public static int toInt(byte[] bytes) {
        Asserts.assertByteArrayLength(bytes, 4);
        return bytes[3] << 24 | (bytes[2] & 0xff) << 16 | (bytes[1] & 0xff) << 8 | (bytes[0] & 0xff);
    }
}
