package rawnet.code.net.core;

import java.io.*;
import java.lang.reflect.Constructor;
import java.util.Base64;

/**
 * Утилиты для полиморфной сериализации {@link Persistable} по имени класса.
 *
 * <p>Формат: {@code UTF className} + {@code instance.save(out)}.
 * Восстановление: {@code Class.forName}, конструктор без аргументов,
 * {@code instance.load(in)}.
 */
public final class Persistables {

    private Persistables() {}

    /** Сохранить объект: имя класса + состояние. */
    public static void save(DataOutputStream out, Persistable p) throws IOException {
        out.writeUTF(p.getClass().getName());
        p.save(out);
    }

    /** Загрузить объект: имя класса + состояние. */
    public static Persistable load(DataInputStream in) throws IOException {
        String className = in.readUTF();
        Class<?> clazz;
        try {
            clazz = Class.forName(className);
        } catch (ClassNotFoundException e) {
            throw new IOException("Неизвестный класс Persistable: " + className, e);
        }
        if (!Persistable.class.isAssignableFrom(clazz))
            throw new IOException("Класс не реализует Persistable: " + className);
        try {
            Constructor<?> ctor = clazz.getConstructor();
            Persistable p = (Persistable) ctor.newInstance();
            p.load(in);
            return p;
        } catch (NoSuchMethodException e) {
            throw new IOException("Нет конструктора без аргументов у " + className, e);
        } catch (ReflectiveOperationException e) {
            throw new IOException("Ошибка создания " + className, e);
        }
    }

    /** Типизированная обёртка: {@code load(in)} + каст. */
    @SuppressWarnings("unchecked")
    public static <T extends Persistable> T load(DataInputStream in, Class<T> type) throws IOException {
        Persistable p = load(in);
        if (!type.isInstance(p))
            throw new IOException("Ожидался " + type.getName() + ", получен " + p.getClass().getName());
        return (T) p;
    }

    // =====================================================================
    // ==================== base64 (Persistables) =========================
    // =====================================================================

    private static String encode(Persistable p) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            Persistables.save(out, p);
        }
        return Base64.getEncoder().encodeToString(bytes.toByteArray());
    }

    private static <T extends Persistable> T decode(String b64, Class<T> clazz) throws IOException {
        byte[] bytes = Base64.getDecoder().decode(b64);
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
            return Persistables.load(in, clazz);
        }
    }
}