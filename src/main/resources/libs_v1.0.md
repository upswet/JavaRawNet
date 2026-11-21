# rawnet

Учебная/исследовательская нейросетевая библиотека на чистой Java. Без внешних зависимостей, кроме Lombok (генерация boilerplate) и SLF4J (логи). Реализует ручной autograd (forward/backward для каждого слоя), собственную бинарную сериализацию и полный стек для supervised- и RL-задач: MLP, CNN, RNN/LSTM/GRU, BiRNN, self-/cross-/N-input/QKV-attention, эмбеддинги (dense и sparse AdamW), нормализации (LayerNorm/InstanceNorm), пулинг по токенам, маски последовательностей, а также RL-тренеры (DQN, Double DQN, REINFORCE, PPO, tabular Q-learning).

Библиотека задумана как **учебная**: каждый класс — с развёрнутым javadoc, где объяснено, *зачем* это, *какие* у этого сильные/слабые стороны, *когда* применять и *как* это соотносится с аналогами из PyTorch/TensorFlow. При этом API и внутренние контракты — «взрослые»: явные интерфейсы, инварианты, проверки на этапе сборки графа, отсутствие глобального состояния (кроме `ThreadLocalRandom` для инициализации/dropout).

---

## Содержание

- [Философия](#философия)
- [Структура пакетов](#структура-пакетов)
- [Быстрый старт](#быстрый-старт)
- [Модели: `Sequence` и `Functional`](#модели-sequence-и-functional)
- [Слои](#слои)
- [Спецификации (`NetSpec`)](#спецификации-netspec)
- [Маски последовательностей](#маски-последовательностей)
- [N входов и merge-операции](#n-входов-и-merge-операции)
- [Функции активации](#функции-активации)
- [Функции потерь](#функции-потерь)
- [Метрики](#метрики)
- [Оптимизаторы](#оптимизаторы)
- [Датасеты и стримы](#датасеты-и-стримы)
- [Сериализация](#сериализация)
- [Reinforcement Learning](#reinforcement-learning)
- [Соглашения и инварианты](#соглашения-и-инварианты)
- [Ограничения](#ограничения)

---

## Философия

1. **Явные контракты вместо магии.** `Layer` — forward/backward/zeroGrad/update/save/load; `Model` — сеть как блок; `NInputModel` — N входов; `Merge` — операция слияния; `Persistable` — бинарная сериализация. Всё, что можно проверить на этапе сборки, проверяется на этапе сборки (формы, число входов, совместимость merge, наличие масок).
2. **Один класс — одна ответственность.** `DenseLayer` не содержит активации; `ActivationLayer` не содержит весов; `ConvLayer` возвращает логиты; `BatchNorm`/`LayerNorm`/`InstanceNorm` — отдельные слои. Это позволяет собирать произвольные топологии и переиспользовать блоки.
3. **Граф как first-class citizen.** `Functional` — DAG, а не цепочка. Узлы: `INPUT`, `LAYER`, `MERGE`. Топологическая сортировка, мемоизация `NetSpec`, forward/backward по топологии, автоматическая маршрутизация по контрактам (`NInput*` → `TwoInput*` → `MaskConsumer` → `Layer`).
4. **Маска — не костыль.** `float[T]` (1 — валидный токен, 0 — PAD) живёт в `Node.mask` (transient) и распространяется по графу по правилам `MaskSource`/`MaskConsumer`/`MaskTransform`/`MaskKiller`. `EmbeddingLayer` — нейтрален (маска идёт мимо), RNN/LSTM/GRU — «замораживают» состояние на PAD, attention — исключает PAD-ключи из softmax и обнуляет `dQ`/`dK`/`dV` для PAD.
5. **Сериализация без `Serializable`.** Ручной `save`/`load` в `DataOutputStream`/`DataInputStream`. `Persistables.save/load` пишет `UTF className` перед телом объекта и восстанавливает его через `Class.forName` + конструктор без аргументов. Это стабильнее и переносимее, чем Java-сериализация, и позволяет полиморфно сохранять `Layer`/`Loss`/`Activate`/`Optimizer`/`Metric`.
6. **Состояние оптимизатора — не часть модели.** `AdamW`/`Momentum`/`NAG` сохраняют только гиперпараметры. После `load` моменты `m/v` и velocity начинаются с нуля. Для инференса это не нужно; для продолжения обучения — стандартная семантика «сброс оптимизатора».
7. **Дублирование ради читаемости.** `softmaxRow` есть в каждом attention-слое; `linear`/`accumLinearGrad` — в каждом attention-слое; `clipByNorm` — в каждой RNN-ячейке. Это осознанно: жёсткая связка с локальными `T`/`d`/`hidden` и желание избежать «утилитарного» слоя, который скрывает зависимости.

---

## Структура пакетов

```
rawnet.code
├── dataset/              Sample, NInputSample, SampleDataset<T>, SampleStream<S>
├── net/
│   ├── TensorShape.java  flat / seq / cnn + save/load
│   ├── core/
│   │   ├── Persistable, Persistables
│   │   ├── activate/     Activate: Linear, Sigmoid, Tanh, LeakyRelu, Gelu, Silu, Mish
│   │   ├── loss/         Loss: Mse, Mae, BCE, BinaryFocalLoss, SoftmaxCrossEntropy, SequenceCrossEntropy
│   │   ├── metric/       Metric: Accuracy, MAE, RMSE, R2, Precision, Recall, F1, TokenAccuracy, LossMetric
│   │   └── optimizator/  Optimizer: SGD, Momentum, NAG, AdamW
│   ├── layers/
│   │   ├── Layer.java    базовый класс + Dto
│   │   ├── contract/     NInputLayer, NInputMaskedLayer, Mask{Consumer,Transform,Source,Killer}, IndexConsumer
│   │   ├── core/         Dense, Activation, Dropout, Embedding, LayerNorm, InstanceNorm, TimeDistributedDense, PositionalEncoding
│   │   ├── cnn/          ConvLayer
│   │   ├── rnn/          ElmanRnn, Gru, Lstm, BidirectionalLayer, BidirectionalRnnLayer (deprecated)
│   │   ├── attention/    MultiHeadAttention, CrossAttention, NInputAttention, QKVAttention, NegativeSampling
│   │   ├── mask/         MaskingLayer (Source), TokenPoolingLayer (Consumer+Killer)
│   │   └── structure/    ModelLayer, ReshapeLayer, FlattenLayer, SliceLayer
│   ├── model/
│   │   ├── Model.java    контракт сети (forward + слоевой контракт + save/load)
│   │   ├── NInputModel.java, TrainableModel.java
│   │   ├── sequence/     Sequence (линейная цепочка слоёв)
│   │   └── functional/   Functional (DAG) + Merge + spec/
│   └── rl/
│       ├── Environment.java, Categorical.java
│       └── training/     RlTrainer, RlTraining, RlMetrics,
│                         QLearningTrainer, DqnTrainer, DoubleDqnTrainer,
│                         ReinforceTrainer, PpoTrainer
```

---

## Быстрый старт

### MLP через `Sequence`

```java
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.*;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.dataset.*;

import java.util.List;

List<Layer.Dto> layers = List.of(
    new Sequence.InputDto(784),
    new DenseLayer.Dto(TensorShape.flat(128)),
    new ActivationLayer.Dto(new LeakyRelu()),
    new DropoutLayer.Dto(0.2f),
    new DenseLayer.Dto(TensorShape.flat(10))     // логиты, БЕЗ активации
);
Model net = new Sequence(layers);
net.train(epochs, dataset, new SoftmaxCrossEntropy(), new AdamW(1e-3), batchSize, 3, metrics, null);
```

### Та же сеть через `Functional` + `NetSpec`

```java
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.functional.spec.*;

NetSpec net = new LayerSpec(new InputSpec(784),
        new DenseLayer.Dto(TensorShape.flat(128)),
        new ActivationLayer.Dto(new LeakyRelu()),
        new DropoutLayer.Dto(0.2f),
        new DenseLayer.Dto(TensorShape.flat(10)));
Functional model = net.build();
```

### ResNet-блок

```java
NetSpec x   = new NamedSpec("x", new InputSpec(784));
NetSpec h   = new LayerSpec(x,
        new DenseLayer.Dto(TensorShape.flat(64)),
        new ActivationLayer.Dto(new LeakyRelu()));
NetSpec res = new AddSpec(h, new RefSpec("x"));
Functional model = res.build();
```

### RNN с маской

```java
List<Layer.Dto> layers = List.of(
    new Sequence.InputDto(T),
    new MaskingLayer.Dto(0),                    // padIndex = 0
    new EmbeddingLayer.Dto(vocab, 128),         // маска идёт мимо
    new BidirectionalLayer.Dto(T, 64, true, new LstmLayer.Dto(T, 64, true)),
    new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN),
    new DenseLayer.Dto(TensorShape.flat(2))
);
Model net = new Sequence(layers);
```

### Self-attention

```java
NetSpec net = new LayerSpec(new InputSpec(TensorShape.seq(64, 128)),
        new MultiHeadAttentionLayer.Dto(8, /* causal = */ false),
        new LayerNormLayer.Dto(),
        new DenseLayer.Dto(TensorShape.flat(10)));
Functional model = net.build();
```

### Cross-attention в `Functional`

```java
Functional.Builder b = Functional.builder(
        TensorShape.seq(16, 128),   // Q
        TensorShape.seq(32, 128));  // KV
Functional.Node q  = b.input();
Functional.Node kv = b.kvInput();
Functional.Node att = b.layerN(
        new CrossAttentionLayer.Dto(8, 128, 128, 32), q, kv);
Functional model = b.output(att);
```

### RL: DQN на CartPole

```java
Environment env = new CartPole();
Functional qNet = Functional.builder(env.observationSize())
        .dense(b.input(), 64).activation(b.getNode("dense0"), new LeakyRelu())
        .build();
// ... собрать qNet/targetNet, создать DqnTrainer, обернуть в RlTraining
RlTrainer trainer = new DqnTrainer(env, qNet, targetNet, new AdamW(1e-3),
        0.99f, 1.0f, 0.05f, 10_000, 42L,
        50_000, 64, 500, 500, 1.0f);
new RlTraining(trainer, 500).train();
```

---

## Модели: `Sequence` и `Functional`

### `Sequence`

Линейная цепочка слоёв. Задаётся списком `Layer.Dto` (первый — `Sequence.InputDto(shape)`). `forward` идёт по слоям, `backward` — в обратном порядке. Это самый простой способ собрать MLP/CNN/RNN-стек.

- `Model.forward(float[], boolean)` — одиночный пример;
- `Model.train(...)` / `test(...)` — батчевое обучение с перемешиванием, zeroGrad → forward+backward → update, периодическая валидация на `testSamples`;
- `Model.backward / zeroGrad / update / gradNormSq / scaleGrad` — слоевой контракт, чтобы вкладывать `Sequence` в другую сеть через `ModelLayer`.

### `Functional`

DAG с произвольной топологией. Узлы: `INPUT`, `LAYER`, `MERGE`. Строится через `Functional.Builder`:

- `builder(int | TensorShape | TensorShape... | List<TensorShape>)` — 1, 2 или N входов;
- `b.input()`, `b.kvInput()`, `b.nInput(i)` — корневые узлы;
- `b.dense`, `b.activation`, `b.layer(dto, src)`, `b.layerN(dto, inputs...)`, `b.crossAttention(q, kv, dto)`, `b.model(model, src)`;
- `b.add/mul/sub/max/min/avg/concat` и `b.merge(Merge, nodes...)` — merge-узлы;
- `b.outputs(NetSpec...)` — несколько выходов; `b.output(node)` — финальный.

`Functional` реализует **и** `Model` (для N=1), **и** `NInputModel`:

| `nInputs()` | Доступные методы |
|---|---|
| 1 | `forward`, `backward`, `forwardN`, `backwardN` |
| 2 | `forward2`, `backward2`, `forwardN`, `backwardN` |
| ≥3 | только `forwardN`, `backwardN` |

Неподходящий метод бросает `IllegalStateException` с указанием, что использовать.

**Маршрутизация слоёв в `runGraph`:**

1. `NInputMaskedLayer` — если у всех входов узла есть маска;
2. `NInputLayer` — иначе;
3. `TwoInputMaskedLayer` — если ровно 2 входа и у обоих есть маска;
4. `TwoInputLayer` — если ровно 2 входа;
5. `MaskConsumer` — обычный слой с маской;
6. `Layer` — обычный слой.

Слой не должен реализовывать одновременно `NInputLayer` и `TwoInputLayer` — маршрутизация выбрала бы неоднозначно.

**Merge** (`rawnet.code.net.model.functional.Merge`): `add`, `mul`, `sub`, `max`, `min`, `avg`, `concat`. Каждый — value-объект с `typeId()` + `saveConfig(DataOutputStream)`, сериализуется автоматически. `mul`/`max`/`min` требуют `setLastInputs(...)` перед `forward` (для backward) — `Functional` делает это сам. Для `add`/`mul`/`max`/`min`/`avg` формы входов должны совпадать; для `concat` — совпадать пространственные оси, каналы складываются.

---

## Слои

Все слои — наследники `Layer`:

- `forward(float[], boolean)` — прямой проход (возвращает внутренний буфер, может быть алиасом входа);
- `backward(float[])` — обратный проход, **накапливает** градиенты весов;
- `zeroGrad()`, `update(Optimizer, batchSize)`, `gradNormSq()`, `scaleGrad(scale)` — параметрический контракт;
- `save(DataOutputStream)` / `load(DataInputStream)` — `super.save` пишет `className + inputShape + outputShape`, наследник дописывает своё;
- `Dto.create(TensorShape inputShape)` — единственный «правильный» способ создать слой: прокидывает вход, вычисляет выход, вызывает `allocateBuffers()` и (где есть веса) `initWeights(...)`.

### Core

| Слой | Формат | Комментарий |
|---|---|---|
| `DenseLayer` | `flat → flat` | Xavier/Glorot, `pendingActivate` — подсказка инициализации |
| `ActivationLayer` | поэлементный | Без параметров, кодирует активацию байтом |
| `DropoutLayer` | поэлементный | Inverted dropout, `p ∈ [0, 1)` |
| `EmbeddingLayer` | `flat(T) → seq(T, dModel)` | `IndexConsumer`; dense / sparse AdamW |
| `LayerNormLayer` | поэлементный | По всем признакам примера |
| `InstanceNormLayer` | поэлементный | По каналам (`C` — ведущая ось) |
| `TimeDistributedDenseLayer` | `seq(T, dIn) → seq(T, dOut)` | Одна `W[dOut][dIn]` на все токены |
| `PositionalEncodingLayer` | `seq(T, d) → seq(T, d)` | Синусоидальное PE (Vaswani) |

### RNN

| Слой | Формат | Комментарий |
|---|---|---|
| `ElmanRnnLayer` | `seq(T, f) → seq(T, H)` / `flat(H)` | `h_t = f(Wx·x_t + Wh·h_{t-1} + b)` |
| `GruLayer` | то же | 3 блока: `z, r, n` |
| `LstmLayer` | то же | 4 блока: `i, f, o, g`; `bf` init в 1 |
| `BidirectionalLayer` | `seq(T, f) → seq(T, 2H)` / `flat(2H)` | Обёртка над любым RNN; сохраняет код типа ячейки |
| `BidirectionalRnnLayer` | то же | `@Deprecated`, только `ElmanRnnLayer` |

Все RNN реализуют `MaskConsumer`: на PAD-шагах `h_t = h_{t-1}` (у LSTM ещё `c_t = c_{t-1}`), градиент проходит сквозь, но веса не накапливаются. Локальный `clipGradNorm` на каждом шаге BPTT.

`BidirectionalLayer` реализует `MaskConsumer`; `forwardRnn` получает маску как есть, `backwardRnn` — перевёрнутую.

### Attention

| Слой | Контракт | Входы |
|---|---|---|
| `MultiHeadAttentionLayer` | `MaskConsumer` | 1× `seq(T, dModel)` (self-attention) |
| `CrossAttentionLayer` (TwoInput) | `TwoInputLayer` / `TwoInputMaskedLayer` | Q + KV |
| `CrossAttentionLayer` (NInput) | `NInputLayer` / `NInputMaskedLayer` | Q + KV, функционально то же |
| `NInputAttentionLayer` | `NInputLayer` / `NInputMaskedLayer` | Q + (N−1) KV-групп, у каждой своя `Wk/Wv` |
| `QKVAttentionLayer` | `NInputLayer` / `NInputMaskedLayer` | Q, K, V — готовые; только `Wo` |

Общие свойства:
- scaled dot-product: `softmax(Q·Kᵀ / √dHead)`, `dHead = dModel / heads`;
- маска: PAD-ключи → `-inf` в scores; PAD-запросы → `dQ = 0`; PAD-ключи/значения → `dK = dV = 0`;
- `causal` опционально (`MultiHeadAttentionLayer`, `QKVAttentionLayer`);
- `softmaxRow` с защитой от полностью PAD/causal-пустой строки (все `-inf` → нули, не NaN).

### Mask

| Слой | Контракт | Формат |
|---|---|---|
| `MaskingLayer` | `MaskSource` | `flat(T)` (индексы токенов, `padIndex`), строит `float[T]` маску |
| `TokenPoolingLayer` | `MaskConsumer` + `MaskKiller` | `seq(T, D) → flat(D)`; режимы `LAST_VALID` / `MEAN` / `MAX` / `FIRST` / `ATTENTION` |

`TokenPoolingLayer.Mode` сериализуется `ordinal()` — **порядок констант менять нельзя** без миграции формата; новые добавлять в конец.

### Structure

| Слой | Комментарий |
|---|---|
| `ModelLayer` | обёртка над готовой `Model`; флаг `trainable` (transfer learning) |
| `ReshapeLayer` | меняет `TensorShape`, данные копирует как есть; теряет маску |
| `FlattenLayer` | `ReshapeLayer` с выходом `flat(inputShape.size())` |
| `SliceLayer` | срез по `FLAT` / `CHANNEL` / `SEQ`; `CHANNEL` сохраняет маску, `SEQ` режет, `FLAT` теряет |

### CNN

| Слой | Формат | Комментарий |
|---|---|---|
| `ConvLayer` | `cnn(C,H,W) → cnn(outC, outH, outW)` | 2D-свёртка без активации; параллелизация по каналам при `MACs > 200_000` |

---

## Спецификации (`NetSpec`)

`NetSpec` — декларативное описание подсети; живёт только на этапе сборки. `build()` — корень сети (только у `InputSpec`), `Functional.Builder.outputs(NetSpec...)` — несколько выходов, `Builder.buildSpec(spec)` — низкоуровневый вызов с мемоизацией.

| Класс | Назначение |
|---|---|
| `InputSpec(int \| TensorShape)` | корень; одноразовый в билдере (повторное использование — `IllegalStateException`) |
| `NamedSpec(name, inner)` | регистрирует корневой узел `inner` под именем |
| `RefSpec(name)` | ссылка на ранее зарегистрированный узел; не может быть корнем |
| `MultiSpec(inner, names...)` | строит `inner` и отдаёт узлы по именам; проверка «имя зарегистрировано внутри» через поколение билдера |
| `LayerSpec(prev, steps...)` | шаги — `Layer.Dto` или `NetSpec`; пробрасывает `pendingActivate` между `Dense/Conv` и следующим `Activation` |
| `DenseSpec(units, act, prev)` | сахар `Dense + Activation` |
| `ModelSpec(prev, model, trainable)` | готовая модель как блок |
| `AddSpec` / `MulSpec` / `SubSpec` / `MaxSpec` / `MinSpec` / `AvgSpec` / `ConcatSpec` | merge-операции декларативно |

Мемоизация по идентичности (`IdentityHashMap`): два структурно равных `LayerSpec` — **разные** spec'и, дадут два независимых подграфа. Один и тот же экземпляр — один узел (общий ствол, residual).

---

## Маски последовательностей

Маска — `float[T]` (1 — валидный токен, 0 — PAD), живёт в `Node.mask` (transient), не сериализуется.

| Контракт | Что делает |
|---|---|
| `MaskSource` | создаёт маску (`lastMask()`) |
| `MaskConsumer` | читает маску, не меняет (`forwardMasked` / `backwardMasked`) |
| `MaskTransform` | меняет маску согласованно с данными (`transformMask`) |
| `MaskKiller` | «съедает» маску (после него `mask = null`) |

Правила:
- `EmbeddingLayer` — нейтрален, маска идёт мимо;
- RNN/LSTM/GRU — «замораживают» состояние на PAD, градиент проходит сквозь, веса не накапливаются;
- attention — PAD-ключи исключаются из softmax (`-inf`), PAD-запросы не обучают `Wq`, PAD-ключи/значения не обучают `Wk/Wv`;
- merge — все входы должны иметь одинаковую маску, на выход идёт маска первого;
- `NInput*`/`TwoInput*` — маска выхода = маска **первого** входа (Q).

---

## N входов и merge-операции

`Functional` поддерживает 1, 2 и N≥3 входов. Для 2 входов — Q + KV (`b.kvInput()`, `b.crossAttention(...)`, `forward2`/`backward2`). Для N — `b.nInput(i)`, `b.layerN(dto, inputs...)`, `forwardN(float[][], boolean)` / `backwardN(float[]) → float[][]`.

`Merge`:
- `add`/`mul`/`sub`/`max`/`min`/`avg` — формы входов должны совпадать;
- `concat` — пространственные оси совпадают, каналы складываются;
- `mul`/`max`/`min` требуют `setLastInputs(...)` перед `forward` (для backward) — `Functional` вызывает сам.

---

## Функции активации

`Activate extends Persistable` — `apply(input, output)`, `derivative(input, output, deriv)` (второй аргумент — кэш прямого прохода; нужен `Sigmoid`/`Tanh`, чтобы не пересчитывать).

| Класс | Формула / комментарий |
|---|---|
| `Linear` | тождество, `deriv = 1` |
| `Sigmoid` | `1/(1+e^{-x})`, `deriv = out(1-out)` |
| `Tanh` | `tanh(x)`, `deriv = 1-out²` |
| `LeakyRelu(alpha=0.01)` | `x > 0 ? x : alpha*x` |
| `Gelu(approximate)` | `x·Φ(x)`; точная (`erf`) или tanh-приближение |
| `Silu` (= Swish) | `x·σ(x)` |
| `Mish` | `x·tanh(softplus(x))` |

`ActivationLayer` кодирует активацию байтом: `1=LeakyRelu(alpha)`, `2=Linear`, `3=Sigmoid`, `4=Tanh`, `5=Gelu(approx)`, `6=Silu`, `7=Mish`. Неизвестный тип — `IOException`.

---

## Функции потерь

`Loss extends Persistable` — `loss(predicted, target)` и `derivative(predicted, target, deriv)` (in-place, длина = длине `predicted`).

| Класс | Ожидает на входе | Комментарий |
|---|---|---|
| `Mse` | любые | `dL/dp = p - t` |
| `Mae` | любые | `dL/dp = sign(p - t)`, 0 в нуле |
| `BinaryCrossEntropy` | **вероятности** (`p = σ(z)`) | Пара строго с `Sigmoid` на выходе: `dL/dp = (p-t)/(p(1-p))`, после σ-производной даёт `p-t` |
| `BinaryFocalLoss(alpha=0.25, gamma=2)` | вероятности | То же — `dL/dp`, композиция с `Sigmoid` |
| `SoftmaxCrossEntropy` | **логиты** | `dL/dz = p - t`, max-trick, `logZ`; выходной слой — без активации |
| `SequenceCrossEntropy` | логиты `[T, vocab]`, `target[t]` — индекс | Реализует `MaskAcceptLoss`, усреднение по `valid`, не по `T` |

`MaskAcceptLoss` — отдельный интерфейс (не часть `Loss`), чтобы `ModelTraining` мог по `instanceof` вызвать `lossMasked`/`derivativeMasked`.

---

## Метрики

`Metric extends Persistable` — `update(prediction, target)`, `result()`, `reset()`, `name()`. Метрики **stateful**; `reset()` обязателен перед каждой эпохой/тестом.

| Класс | Комментарий |
|---|---|
| `Accuracy(threshold=0.5)` | `len==1` — порог; `len>1` — argmax |
| `Mae`, `Rmse`, `R2` | регрессия; `R2` — по всем элементам всех батчей |
| `Precision`, `Recall`, `F1` | macro; `ClassificationStats` копит `tp/fp/fn` по классам; `len==1` — 2 класса |
| `TokenAccuracy` | per-token argmax по `[T, vocab]`, реализует `MaskAcceptMetric` |
| `LossMetric(Loss)` | обёртка над `Loss`; сериализует `Loss` через `Persistables` |

`MaskAcceptMetric` — отдельный интерфейс, симметрично `MaskAcceptLoss`.

---

## Оптимизаторы

`Optimizer extends Persistable` — `updateWeights(float[][] w, float[][] g, int bs)`, `updateBiases(float[] b, float[] g, int bs)`, дефолтный `update(w, g, b, bg, bs)`.

| Класс | Состояние | `save`/`load` |
|---|---|---|
| `SGD(lr)` | нет | `lr` |
| `Momentum(lr, momentum)` | velocity (`IdentityHashMap` по ключу-массиву) | `lr, momentum`; velocity сбрасывается |
| `NAG(lr, momentum)` | velocity | `lr, momentum`; velocity сбрасывается |
| `AdamW(lr=1e-3, β1=.9, β2=.999, ε=1e-8, wd=.01)` | `m/v` + счётчики шагов | 5 гиперпараметров; `m/v` сбрасываются |

Weight decay в `AdamW` применяется **только к весам**, не к bias. Для `Momentum`/`NAG` velocity хранится по `IdentityHashMap` (ключ — сам массив параметров); после `load` velocity начинается с нуля.

---

## Датасеты и стримы

- `Sample` — один пример: `float[] inputs`, `float[] targets`.
- `NInputSample` — `float[][] inputs` + `float[] targets`; `nInputs() == inputs.length`, порядок = порядок в `forwardN`.
- `SampleDataset<T>` — `trainSamples` + `testSamples`, generic по типу примера; `SampleDataset.split(List<T>, float procent)` — shuffle + резка.
- `SampleStream<S>` — ленивый источник; `open()` → `Iterator<S>`, `size()` (по умолчанию -1), `getTestSamples()` **обязателен** (не default — чтобы не забыть и не потерять метрики на test). Фабрики: `of`, `fromIterator`, `fromSupplier`; `stream()` — `Stream<S>`.

---

## Сериализация

- `Persistable` — `save(DataOutputStream)` / `load(DataInputStream)`. **Имя класса не пишется** — его пишет вызывающий.
- `Persistables.save(out, p)` — пишет `UTF className` + `p.save(out)`.
- `Persistables.load(in)` — читает `UTF className`, `Class.forName`, конструктор без аргументов, `load`.
- `Persistables.load(in, Class<T>)` — типизированная обёртка с проверкой `isInstance`.
- `Model.save(Path)` / `Model.load(Path)` — на уровне модели (UTF-строка типа `Sequence`/`Functional` + тела).
- `Model.load(DataInputStream)` делает `mark`/`reset`, чтобы тип прочитали загрузчики (`Sequence.load` / `Functional.load`), а не сам `Model.load`.

`ActivationLayer.writeActivate/readActivate` — статический кодек активаций (байт-код + параметры). `BidirectionalLayer` использует собственный код типа ячейки: `0=Elman`, `1=GRU`, `2=LSTM`, `-1=OTHER`. Для `CELL_OTHER` при `load` пользователь обязан задать `cellDto` до вызова `load` (иначе `IOException`).

**Что НЕ сохраняется (намеренно):**

| Что | Почему |
|---|---|
| `AdamW.m/v`, `Momentum/NAG.velocity` | нужно только для продолжения обучения; для инференса бесполезно |
| `Node.mask`, `Node.output`, `Node.deltaAccum` | transient-буферы прямого/обратного прохода |
| Sparse-состояние `EmbeddingLayer` (`mW/vW/step`) | сбрасывается в `null` при `expand`/`expandAndRemap` и при `load` |
| `dropout`-маски, RNN-кэши (`hCache`, `zCache`, ...) | пересоздаются в `allocateBuffers` |
| `pendingActivate` в `Dto` | только для инициализации, после `load` не нужно |

**Атомарность:** `Model.save(Path)` не атомарна — при падении посередине файл останется частично записанным. Если нужна атомарность, пишите во временный файл и делайте `Files.move` с `ATOMIC_MOVE`.

---

## Reinforcement Learning

Инфраструктура — `rawnet.code.rl` + `rawnet.code.rl.training`.

### `Environment`

```java
public interface Environment {
    float[] reset();
    StepResult step(int action);   // action ∈ [0, numActions())
    int numActions();
    int observationSize();
    default void render() {}

    record StepResult(float[] observation, float reward, boolean done) {}
}
```

Эпизоды: `reset()` → `step(a)` → ... → `done == true` → `reset()`. Действия — дискретные `int` (для непрерывного RL нужен отдельный интерфейс). Наблюдение — `float[]`, ровно то, что принимает `Functional`.

### `Categorical`

- `sample(logits, Random)` — inverse-CDF по `softmax(logits)`, max-trick;
- `logProb(logits, action)` — `log softmax(logits)[action]`;
- `softmax(logits)` — для инференса/отладки.

### `RlTrainer` / `RlTraining` / `RlMetrics`

- `RlTrainer` — `trainIteration()`, `lastMetrics()`. Одна итерация — эпизод (REINFORCE, Q-learning, DQN) или rollout (PPO).
- `RlMetrics(avgReward, avgEpisodeLength, policyLoss, valueLoss, entropy, episodes)`; `RlMetrics.reinforce(...)` — удобный конструктор без `valueLoss` (`NaN` = «не применимо»).
- `RlTraining(RlTrainer, maxIterations, window, patience, minDelta, logEvery)` — цикл, окно средней награды, early stopping, NaN/Inf-защита. **`valueLoss == NaN` — не расходимость** (у REINFORCE нет critic); проверяется только `isInfinite`.

### Тренеры

| Класс | Что делает | Особенности |
|---|---|---|
| `QLearningTrainer(env, alpha, gamma, ε, ...)` | tabular Q-learning | только для `observationSize() ≤ 1024`; `observation` — one-hot |
| `DqnTrainer(env, qNet, targetNet, opt, γ, ε, ..., bufferSize, batchSize, warmup, syncSteps, huberDelta)` | DQN | replay buffer (кольцевой `ArrayList` + `bufferPos`), target network, Huber loss, ε-greedy с линейным decay |
| `DoubleDqnTrainer` | Double DQN | переопределяет **только** `trainBatch()`: `a* = argmax qNet(s')`, `target = r + γ · targetNet(s')[a*]` |
| `ReinforceTrainer(env, policy, opt, γ, entropyBonus, clipEps, nEpochs, miniBatch, seed)` | REINFORCE + PPO-style clip | gradient clipping по L2-норме, защита `0·log(0)` в энтропии, нормализация advantage |
| `PpoTrainer(env, actor, critic, actorOpt, criticOpt, γ, λ, clipEps, entropyBonus, rolloutSteps, nEpochs, miniBatch, seed)` | PPO + GAE | clip objective, critic MSE, нормализация advantage, `nEpochs` проходов по rollout |

**Общие соглашения:**
- `zeroGrad()` перед накоплением градиентов в батче; `update(optimizer, n)` — в конце.
- DQN `syncTarget()` — через `ByteArrayOutputStream` + `Functional.save/load`; после sync — probe-forward на фиксированном входе, `|online - target| < 1e-6` (иначе `IllegalStateException`: слой не сохраняет/не грузит веса).
- DQN `sampleBatch(k)` — с возвращением, O(k), без копирования всего буфера.
- PPO/REINFORCE: `oldLogits` восстанавливаются **до** обновления (в `training = false`, чтобы избежать dropout), clip `ratio = π_new / π_old ∈ [1-ε, 1+ε]`.

### Пример DQN на CartPole

```java
Environment env = new CartPole();

Functional.Builder b = Functional.builder(env.observationSize());
Functional.Node h = b.layer(new DenseLayer.Dto(64), b.input());
Functional.Node a = b.layer(new ActivationLayer.Dto(new LeakyRelu()), h);
Functional.Node q = b.layer(new DenseLayer.Dto(env.numActions()), a);
Functional qNet = b.output(q);

Functional targetNet = Functional.load(/* сохранить qNet и загрузить копию */);

Optimizer opt = new AdamW(1e-3f);
RlTrainer trainer = new DqnTrainer(env, qNet, targetNet, opt,
        /*γ*/ 0.99f, /*εStart*/ 1.0f, /*εEnd*/ 0.05f, /*εDecaySteps*/ 10_000,
        /*seed*/ 42L,
        /*bufferSize*/ 50_000, /*batchSize*/ 64,
        /*warmupSteps*/ 500, /*targetSyncSteps*/ 500,
        /*huberDelta*/ 1.0f);
new RlTraining(trainer, /*maxIterations*/ 500, /*window*/ 50, /*patience*/ 20,
               /*minDelta*/ 0.05f, /*logEvery*/ 10).train();
```

---

## Соглашения и инварианты

### Формы (`TensorShape`)

- `flat(n)` — вектор длины `n`;
- `seq(T, d)` — последовательность из `T` векторов длины `d`;
- `cnn(C, H, W)` — CNN-тензор `C×H×W`;
- `cnn(C, H, W, T)` — CNN + sequence (для video / 3D).

`TensorShape` умеет `save`/`load`, `equals`, `sameSpatialForConcat`, `size()`, `channels()`, `height()`, `width()`, `seqLen()`, `isFlat()`, `isSequence()`, `isCnn()`.

### `Layer`

- `forward` возвращает внутренний буфер; вызывающий **не должен** рассчитывать на отдельный массив и не должен его модифицировать.
- `backward` вызывается **только** после `forward` для того же примера и **накапливает** градиенты весов.
- `super.save(out)` пишет `className + inputShape + outputShape`; наследник дописывает своё. `super.load(in)` читает формы; наследник — своё, затем обычно `allocateBuffers()`.
- `Dto.create(inputShape)` — единственный способ создать слой «правильно»: прокидывает вход, вычисляет выход, вызывает `allocateBuffers()` и (где есть веса) `initWeights(...)`.

### `Model`

- `forward(input, training)` — одиночный пример; батч организуется на уровне `train`/`test`.
- Слоевой контракт (`backward`/`zeroGrad`/`update`/`gradNormSq`/`scaleGrad`) — для вложения через `ModelLayer`. По умолчанию бросает `UnsupportedOperationException`; `Sequence` и `Functional` его поддерживают.
- `save(Path)` / `load(Path)` — на уровне модели; `save(DataOutputStream)` / `load(DataInputStream)` — для вложенных моделей.

### `Functional`

- `nInputs() == 1` → `forward`/`backward`; `== 2` → ещё `forward2`/`backward2`; `≥ 3` → только `forwardN`/`backwardN`. Неподходящий метод → `IllegalStateException`.
- Маршрутизация: `NInputMaskedLayer` → `NInputLayer` → `TwoInputMaskedLayer` → `TwoInputLayer` → `MaskConsumer` → `Layer`. Слой не должен реализовывать одновременно `NInputLayer` и `TwoInputLayer`.
- Маска выхода `NInput*`/`TwoInput*` — маска **первого** (Q) входа; у merge все входы должны иметь одинаковую маску.
- `output(node)` проверяет, что все объявленные входы использованы; незадействованный вход — `IllegalStateException`.
- `builder(...).output(...)` очищает `pendingActivateForNext` у всех узлов — `ActivationLayer` до merge не «протекает» в `Dense`/`Conv` после merge.

### `NetSpec`

- Мемоизация по идентичности: один и тот же экземпляр — один узел (общий ствол); два структурно равных — два независимых подграфа.
- `InputSpec` — одноразовый в билдере; для переиспользования — `NamedSpec` + `RefSpec`.
- `MultiSpec` проверяет «имя зарегистрировано внутри `inner`» через `currentGeneration`/`generationOf`; имена, зарегистрированные до вызова, отвергаются.
- `LayerSpec` пробрасывает `pendingActivate` только по непрерывным `DtoStep`; `SpecStep` разрывает цепочку.

### `Loss` / `Activate` / `Metric`

- `Loss.derivative(predicted, target, deriv)` — **in-place**, длина = длине `predicted`.
- `Activate.derivative(input, output, deriv)` — `output` — кэш `apply`, не пересчитывать.
- «Композитные» лоссы (`BinaryCrossEntropy`, `BinaryFocalLoss`) отдают `dL/dp`; строго с `Sigmoid` на выходе. `SoftmaxCrossEntropy` — уже `dL/dz`; выходной слой — без активации.
- Метрики stateful: `reset()` обязателен перед каждой эпохой/тестом. `ClassificationStats.reset` обнуляет `tp/fp/fn`, но **не** уменьшает `numClasses` — массивы переиспользуются.
- `R2` считает по **всем** элементам всех `update`, а не по примерам.

### `Optimizer`

- Ключ в `IdentityHashMap` — сам массив параметров; после `load` состояние `m/v/velocity` пустое.
- `AdamW.weightDecay` — только к весам, не к bias.
- `Optimizer.update(w, g, b, bg, bs)` — общий контракт; слои без bias (`EmbeddingLayer`) вызывают только `updateWeights`.

### `Persistables`

- `save(DataOutputStream)` **не** пишет имя класса; его пишет вызывающий.
- `load(DataInputStream)` читает ровно то, что записал `save`, в уже созданный экземпляр.
- Требуется конструктор без аргументов (для `Class.forName`).
- `Persistables.load(in, Class<T>)` — типизированная обёртка с проверкой `isInstance`.
- `LossMetric` сериализует `Loss` через `Persistables`, поэтому `Loss` **обязан** иметь конструктор без аргументов (у всех текущих — есть).

### `SampleStream`

- `getTestSamples()` обязателен к реализации (не default) — чтобы пользователь **не забыл** и не потерял метрики на test.
- `size()` по умолчанию `-1` (неизвестно).

---

## Ограничения

Библиотека учебная, и это накладывает ограничения:

- **CPU, single-threaded** для большинства слоёв (исключение — `ConvLayer` при `MACs > 200_000` распараллеливается по каналам). Нет GPU/CUDA, нет `BLAS`/`MKL`.
- **`float[]`, а не многомерные тензоры**; `TensorShape` — метаданные, не тип данных. Слои знают свою раскладку (flat / seq / cnn) и проверяют её на этапе `create`.
- **Батч — снаружи**: `forward` работает с одним примером; `Model.train` организует цикл, `zeroGrad` → forward+backward для каждого примера → `update(optimizer, batchSize)`. Это упрощает отладку, но медленнее, чем batched-реализация.
- **Дискретный RL** (`Environment.step(int)`). Для непрерывных действий нужен отдельный интерфейс.
- **Tabular Q-learning** — только для `observationSize() ≤ 1024` (защита от OOM).
- **DQN `sampleBatch`** — с возвращением (`rnd.nextInt(size)` на каждую позицию); приоритизированный replay (`PER`) не реализован.
- **PPO** — упрощённый: нет отдельного value-clip, нет adaptive KL, нет GAE-лямбды как отдельного гиперпараметра (задаётся в конструкторе `PpoTrainer`).
- **`Dropout`** в attention-слоях — поле есть, но **не применяется** (задел на будущее).
- **`Gelu`** — аппроксимация `erf` по Abramowitz & Stegun 7.1.26 (Java < 20); для Java 20+ можно заменить на `Math.erf`.
- **`softmaxRow`** дублируется в каждом attention-слое (осознанно: жёсткая связка с локальными `T`/`d`).
- **`Optimizer` state** не сохраняется — после `load` `AdamW`/`Momentum`/`NAG` начинают с нуля. Для инференса это ок; для продолжения обучения — стандартная семантика «сброс оптимизатора».
- **`Functional` не поддерживает циклы** в графе (DAG, топологическая сортировка; цикл → бесконечный `while` в `Builder.output` — по факту не соберётся, потому что каждый `addNode` добавляет узел в `nodes`; но явной проверки на цикл нет).
- **`Sequence` и `Functional` не поддерживают batch-dim**: размерности `[B, ...]` нет; каждый пример — отдельный `float[]`.

---

## Лицензия и благодарности

Библиотека написана как учебный проект: каждый класс — с развёрнутым javadoc, ссылками на оригинальные статьи (Vaswani et al. 2017 для Transformer, Cho et al. 2014 для GRU, Hochreiter & Schmidhuber 1997 для LSTM, Van Hasselt et al. 2016 для Double DQN, Lin et al. 2017 для Focal Loss, Gers et al. 2000 для forget-gate bias, He et al. 2015 для He-инициализации, Glorot & Bengio 2010 для Xavier). API вдохновлён PyTorch (`nn.Module`, `nn.MultiheadAttention`, `requires_grad=False`) и Keras (`Sequential`, `Dense`, `LSTM`, `Bidirectional`).

