package rawnet.code.net.layers.contract;

/** Слой, который «съедает» маску: выход — не последовательность (TokenPooling). */
public interface MaskKiller {
    // маркерный интерфейс: после такого слоя mask = null
}
