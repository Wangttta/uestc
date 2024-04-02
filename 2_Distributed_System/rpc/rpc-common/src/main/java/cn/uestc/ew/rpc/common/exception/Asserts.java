package cn.uestc.ew.rpc.common.exception;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;

@UtilityClass
public class Asserts {


    /**
     * 判断给定的字节数组是否达到所要求的长度，没有则抛出异常，否则什么都不做。
     * 字节数组允许为空，但必须指定长度为 0 才不会抛出异常。
     *
     * @param bytes  需要判断长度是否匹配的字节数组
     * @param length 期望的数组长度
     */
    public void assertByteArrayLength(byte[] bytes, int length) {
        if ((bytes == null && length != 0) || (bytes != null && bytes.length != length)) {
            throw new RuntimeException(String.format("AssertsException: Unexpected byte array length {array=%s, length=%d}", Arrays.toString(bytes), length));
        }
    }

    public static void notEmpty(String text, String msg) {
        if (StringUtils.isEmpty(text))
            throw new RuntimeException(msg);
    }

    public static void notNull(Object o, String msg) {
        if (o == null)
            throw new RuntimeException(msg);
    }
}