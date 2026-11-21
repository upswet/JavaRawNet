package rawnet.code.net.layers.contract;

import rawnet.code.net.layers.mask.MaskingLayer;

/**
 * Слой, который принимает на вход <b>индексы токенов</b>
 * ({@code flat(T)}, значения — целые в {@code [0, vocab)}).
 *
 * <p>Единственный потребитель этого интерфейса — проверки совместимости:
 * {@link MaskingLayer} имеет смысл только перед таким слоем (маска
 * строится по индексам, а не по эмбеддингам).
 *
 * <p>Реализуют: {@code EmbeddingLayer}, будущие
 * {@code PositionalEmbeddingLayer}, {@code SharedEmbeddingLayer},
 * {@code TokenTypeEmbeddingLayer} и т.п.
 * <b>Не</b> реализует {@code PatchEmbeddingLayer} (вход — {@code cnn}),
 * {@code DenseLayer}, {@code ConvLayer} и т.п.
 */
public interface IndexConsumer {
}
