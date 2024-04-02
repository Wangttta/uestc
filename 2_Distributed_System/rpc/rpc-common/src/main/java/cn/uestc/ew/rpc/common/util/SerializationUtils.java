package cn.uestc.ew.rpc.common.util;

import com.dyuproject.protostuff.LinkedBuffer;
import com.dyuproject.protostuff.ProtostuffIOUtil;
import com.dyuproject.protostuff.Schema;
import com.dyuproject.protostuff.runtime.RuntimeSchema;
import lombok.experimental.UtilityClass;
import org.objenesis.Objenesis;
import org.objenesis.ObjenesisStd;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 序列化工具，基于 Protostuff
 */
@UtilityClass
public class SerializationUtils {

    /**
     * 缓存 Java 的 {@link Class} 类型与 Protostuff 的 {@link Schema} 类型的对应关系
     */
    private static final Map<Class<?>, Schema<?>> cachedSchema = new ConcurrentHashMap<>();

    /**
     * 实例化工具
     */
    private static final Objenesis objenesis = new ObjenesisStd(true);

    /**
     * 序列化方法，将对象序列化为字节数组
     *
     * @param obj 需要序列化的对象
     * @return 序列化后对应的字节数组
     * @throws IllegalStateException 序列化失败后抛出该异常，可通过其 {@code getCause()} 方法获取原始异常
     */
    @SuppressWarnings("unchecked")
    public static <T> byte[] serialize(T obj) {
        Class<T> cls = (Class<T>) obj.getClass();
        LinkedBuffer buffer = LinkedBuffer.allocate(LinkedBuffer.DEFAULT_BUFFER_SIZE);
        try {
            Schema<T> schema = getSchema(cls);
            return ProtostuffIOUtil.toByteArray(obj, schema, buffer);
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        } finally {
            buffer.clear();
        }
    }

    /**
     * 反序列化方法，将字节数组反序列化为指定对象类型
     *
     * @param data 需要序列化的字节数组
     * @param cls  序列化后的对象类型
     * @return 序列化后的对象，序列化失败会抛出异常而非返回 {@code null}
     * @throws IllegalStateException 序列化失败后抛出该异常，可通过其 {@code getCause()} 方法获取原始异常
     */
    public static <T> T deserialize(byte[] data, Class<T> cls) {
        try {
            T message = objenesis.newInstance(cls);
            Schema<T> schema = getSchema(cls);
            ProtostuffIOUtil.mergeFrom(data, message, schema);
            return message;
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> Schema<T> getSchema(Class<T> cls) {
        Schema<T> schema = (Schema<T>) cachedSchema.get(cls);
        if (schema == null) {
            schema = RuntimeSchema.createFrom(cls);
            cachedSchema.put(cls, schema);
        }
        return schema;
    }
}