Ниже ТЗ именно под статью, а не под “поиграться”. Полностью непридираемой работы не бывает, но если выполнить это ТЗ, основные атаки рецензента будут закрыты: “почему temporal?”, “почему не random split?”, “почему не KS/PSI/MMD?”, “как связан drift score с реальной деградацией модели?”, “что с размером окна?”, “что с категориальными признаками?”, “что с ложными срабатываниями?”, “есть ли статистика, а не один красивый график?”.

Рабочее название:

**Sliced Wasserstein Drift Score for Lightweight Monitoring of Tabular ML under Temporal Distribution Shift**

Основная идея: предложить не новую модель, а лёгкий unsupervised drift-monitoring score для табличных данных. Он считается только по признакам X, не требует новых меток y, работает на временных окнах и должен заранее сигнализировать, что качество уже обученной модели начинает деградировать. Это хорошо попадает в актуальный контекст: TabReD прямо показывает, что реальные табличные данные часто меняются во времени, что time-based split нужен для честной оценки, а random split меняет ранжирование методов; в TabReD также подчёркивается, что GBDT и простые MLP-like модели остаются сильными на таких данных. ([arXiv][1])

## 1. Цель исследования

Цель: проверить, является ли **Sliced Wasserstein Drift Score** дешёвым, устойчивым и практически полезным индикатором temporal distribution shift в табличных ML-задачах.

Фокус не на том, чтобы “победить все методы drift detection”, а на более аккуратном claim:

**SWDS лучше простых marginal drift scores связывается с будущей деградацией качества табличной модели и может использоваться как лёгкий триггер мониторинга/переобучения.**

Это важное сужение. Рецензенту будет труднее придраться, потому что ты не обещаешь универсальный detector всех сдвигов, а проверяешь конкретную operational-задачу: есть модель, есть временные окна, меток в будущем может не быть, нужно понять, ухудшается ли распределение входов.

## 2. Научный вклад, который надо заявлять

В статье должно быть три вклада.

Первый: **практический drift score** для табличных данных на основе sliced Wasserstein distance, который масштабируется лучше, чем полноценный multivariate Wasserstein, и работает на mixed-type tabular features после фиксированного leakage-safe preprocessing.

Второй: **экспериментальный протокол**, который связывает unsupervised drift score не просто с фактом “распределения различаются”, а с реальной деградацией downstream-модели на будущих временных окнах.

Третий: **сравнение с простыми и сильными baseline drift indicators**: KS, PSI, MMD, energy distance, classifier two-sample test, а также с банальным календарным retraining. Это важно, потому что без baseline работа будет выглядеть как “мы посчитали красивую метрику”.

Связь с 2024–2026 такая: TabReD и последующие работы по temporal tabular shift показывают, что temporal shift — не побочная деталь, а центральная проблема табличного ML; TabArena подчёркивает важность стандартизованного и честного протокола оценки, а не просто сравнения моделей на случайных split-ах; Drift-Resilient TabPFN показывает, что temporal distribution shift стал отдельной темой даже для foundation-моделей. ([arXiv][1])

## 3. Формальная постановка

Есть временно упорядоченный табличный датасет:

[
D = {(x_i, y_i, t_i)}_{i=1}^{n}
]

где (x_i) — вектор признаков, (y_i) — целевая переменная, (t_i) — timestamp или surrogate timestamp.

Данные делятся не случайно, а по времени:

[
D_{train} < D_{val} < D_{test}
]

На (D_{train}) обучается модель (f). Далее test-период режется на последовательность окон:

[
W_1, W_2, \ldots, W_T
]

Для каждого окна (W_j) считаются:

1. drift score между reference-окном и текущим окном;
2. downstream-качество модели (f) на этом окне;
3. падение качества относительно reference/validation-периода.

Reference-распределение лучше задавать двумя вариантами:

[
R_0 = D_{train}
]

и

[
R_{recent} = \text{последние } m \text{ временных окон train/val}
]

Первый вариант проверяет drift от исходной обучающей выборки. Второй — более deployment-like: модель обслуживает поток, а reference можно обновлять.

## 4. Определение Sliced Wasserstein Drift Score

Для двух выборок признаков (X_R) и (X_W), после одинакового preprocessing, генерируются (K) случайных направлений (\theta_k), нормированных до единичной длины. Для каждого направления данные проектируются в 1D:

[
z_R^{(k)} = X_R \theta_k,\quad z_W^{(k)} = X_W \theta_k
]

Затем считается одномерная Wasserstein distance между проекциями. Итоговый score:

[
SWDS(X_R, X_W) =
\left(
\frac{1}{K}
\sum_{k=1}^{K}
W_2^2(z_R^{(k)}, z_W^{(k)})
\right)^{1/2}
]

Для вычисления (W_2) в 1D использовать сортировку и quantile interpolation. Если размеры окон разные, не делать тупой subsample как единственный вариант; нужно реализовать два режима:

1. equal-size subsampling с несколькими seed;
2. quantile-grid approximation, например 256 или 512 квантилей.

Основной режим для статьи: quantile-grid approximation, потому что он стабильнее и не привязан к равным размерам окон.

Sliced Wasserstein хорошо обосновывается тем, что он является вычислительно лёгкой проекционной аппроксимацией Wasserstein distance; в свежей работе 2025 года его используют как scalable unsupervised score для anomaly/data selection, именно из-за интерпретируемости через optimal transport и вычислительной лёгкости. ([arXiv][2])

## 5. Главные проверяемые гипотезы

### H1. SWDS лучше marginal drift scores связан с деградацией качества модели

Для каждого датасета, модели и временного окна считаем:

[
\Delta Q_j = Q_{ref} - Q_j
]

где (Q_j) — качество модели на окне (W_j). Для classification: ROC-AUC, PR-AUC, logloss, Brier score. Для regression: RMSE, MAE, (R^2).

Проверка:

[
corr(SWDS_j, \Delta Q_j)
]

Основной критерий: Spearman correlation, потому что связь может быть монотонной, но не линейной.

Сравнить с:

* mean KS по признакам;
* max KS по признакам;
* mean PSI;
* max PSI;
* MMD-RBF;
* energy distance;
* classifier two-sample test.

Гипотеза считается подтверждённой, если SWDS имеет статистически значимо более высокую медианную Spearman correlation по dataset-model парам, чем KS/PSI, и не хуже MMD/C2ST при меньшем времени расчёта.

Минимальный критерий успеха: SWDS входит в top-2 по корреляции минимум на 60% dataset-model пар.

Сильный критерий успеха: SWDS статистически превосходит marginal baselines по Wilcoxon signed-rank test с Holm или BH-FDR correction.

### H2. SWDS раньше детектирует контролируемый drift, чем marginal-тесты

Нужно сделать synthetic drift injection поверх реальных датасетов. Это важно: на реальных данных мы не знаем истинный момент drift, а для detection-задачи нужна ground truth.

Делаем несколько сценариев:

1. mean shift по группе числовых признаков;
2. covariance/correlation shift без сильного изменения marginal distributions;
3. categorical prior shift;
4. label-independent covariate shift;
5. concept-like shift: меняется связь (X \rightarrow y), но marginal X меняется слабо;
6. mixed shift: немного числового, немного категориального, немного missingness.

SWDS должен особенно выигрывать на сценарии covariance/correlation shift, потому что marginal KS/PSI могут его плохо ловить.

Метрики:

* AUROC drift/no-drift;
* AUPRC drift/no-drift;
* detection delay: через сколько окон после начала drift метод срабатывает;
* false alarm rate до drift.

Гипотеза считается подтверждённой, если SWDS даёт меньший detection delay при фиксированном false alarm rate, например 5% или 10%, чем marginal KS/PSI.

### H3. SWDS-triggered retraining лучше календарного retraining при том же бюджете переобучений

Это самая прикладная гипотеза. Не просто “score коррелирует”, а “score полезен для решения”.

Сравнить политики:

1. **No retraining**: модель обучена один раз.
2. **Periodic retraining**: переобучение каждые (p) окон.
3. **KS/PSI-triggered retraining**.
4. **MMD/C2ST-triggered retraining**.
5. **SWDS-triggered retraining**.
6. **Oracle retraining**: переобучение там, где реально произошло заметное падение качества. Это не deployable baseline, а верхняя граница.

При срабатывании trigger модель переобучается на всех прошлых данных или на rolling history последних (h) окон. Оба режима надо проверить.

Метрики:

[
Regret = \sum_j (Q_{oracle,j} - Q_{policy,j})
]

или для loss:

[
Regret = \sum_j (L_{policy,j} - L_{oracle,j})
]

Также считать:

* среднее качество по test-period;
* worst-window quality;
* число retraining events;
* качество на один retraining;
* долю ложных retraining.

Гипотеза считается подтверждённой, если SWDS-triggered policy даёт меньший regret, чем periodic retraining, при равном или меньшем числе переобучений.

### H4. SWDS остаётся стабильным при малом числе проекций

Проверить (K \in {16, 32, 64, 128, 256}).

Метрики:

* ranking stability между окнами;
* correlation stability;
* runtime;
* variance score по разным random seeds.

Гипотеза: (K = 64) или (K = 128) достаточно, чтобы получить стабильный drift score, а дальнейшее увеличение почти не улучшает связь с деградацией качества.

Практический критерий: при (K=64) корреляция с (\Delta Q) падает не более чем на 5% относительно (K=256), а runtime ниже минимум в 2 раза.

### H5. SWDS должен считаться в leakage-safe feature space

Это не гипотеза “для красоты”, а защита от рецензента. Нужно показать, что результат не возникает из-за неправильного preprocessing.

Сравнить три варианта представления:

1. raw leakage-safe preprocessing: impute/scale/encode fitted only on train;
2. model-aware representation: leaf indices / embeddings / predicted probabilities;
3. PCA-compressed representation после train-only preprocessing.

Основной claim лучше строить на варианте 1, потому что он model-agnostic. Остальные — абляции.

Гипотеза: raw leakage-safe SWDS уже полезен; model-aware representation может улучшать связь с quality drop, но теряет универсальность.

## 6. Датасеты

Нужны не случайные UCI-таблицы, а именно temporally ordered tabular datasets. Минимум — 6 датасетов, хорошо — 8–10. Если сделать меньше, рецензент справедливо скажет, что выводы нестабильны.

Основной набор:

**A. TabReD как главный источник.**
TabReD — лучший выбор, потому что это свежий ICLR 2025 benchmark из восьми industry-grade табличных датасетов с time-based split и feature-rich setting. В самой работе сказано, что такие данные ближе к промышленному deployment, а temporal drift и feature-rich datasets недопредставлены в обычных академических бенчмарках. ([arXiv][1])

Использовать все доступные датасеты TabReD, если они скачиваются без проблем. Если какой-то датасет слишком тяжёлый, оставить минимум 5–6, но явно написать критерии исключения: размер, лицензия, невозможность воспроизведения, слишком мало временных окон.

**B. Kaggle/public temporal tabular datasets как внешняя проверка.**
Нужно добавить 2–3 внешних датасета не из TabReD, чтобы работа не выглядела как “мы подогнали метод под один benchmark”.

Подходящие кандидаты:

1. **IEEE-CIS Fraud Detection** — transaction-level fraud classification, есть `TransactionDT`, то есть временной порядок; задача бинарной классификации, сильный дисбаланс, хороший stress-test для PR-AUC.
2. **Rossmann Store Sales** — временная регрессия продаж, много табличных признаков и сезонности.
3. **Bike Sharing / Bike Demand** — лёгкая временная регрессия, хороший sanity-check.
4. **Home Credit / credit-like datasets** — брать только если есть timestamp или корректный surrogate temporal split; иначе не использовать как основной temporal dataset.

Лучше иметь состав:

* 5–8 TabReD datasets;
* IEEE-CIS Fraud Detection;
* Rossmann или Bike Sharing как regression sanity-check;
* один синтетический controlled dataset для чистой проверки drift scenarios.

Итого: 8–11 задач. Это уже достаточно для Q3/Q4.

## 7. Критерии включения датасета

Датасет включается, если выполняются условия:

1. есть timestamp, transaction time, date или официальный time-based split;
2. минимум 10 000 объектов или минимум 20 временных окон после разбиения;
3. есть табличные признаки, а не текст/изображения/аудио;
4. нет явной target leakage после первичной проверки;
5. задача supervised: classification или regression;
6. downstream-модель показывает качество выше trivial baseline.

Датасет исключается, если:

1. временной порядок искусственно восстановлен без обоснования;
2. target leakage невозможно устранить;
3. слишком мало окон для оценки drift-quality correlation;
4. качество модели не лучше константного baseline;
5. данные требуют тяжёлой доменной обработки, не относящейся к статье.

## 8. Разбиение по времени

Нельзя использовать random split в основном эксперименте. Только как дополнительный контроль.

Основной split:

* первые 60% времени: train;
* следующие 20%: validation/calibration;
* последние 20%: test-monitoring stream.

Если в TabReD уже есть официальный split, использовать официальный split. Это важно: TabReD создан именно для time-based evaluation; его авторы показывают, что time-based split может менять ранжирование методов относительно random split. ([arXiv][1])

Test-period режется на окна:

* fixed-count windows: например, по 1000/2000/5000 объектов;
* fixed-time windows: например, день/неделя/месяц, если timestamp нормальный.

Основной режим: fixed-count windows, потому что он стабилизирует дисперсию метрик. Fixed-time windows — robustness check.

Минимальное число test windows: 20. Если меньше — датасет нельзя использовать для основных выводов, только как case study.

## 9. Preprocessing без leakage

Это критически важно.

Все preprocessing-операции должны fit-иться только на train:

* числовые признаки: median imputation + standard scaling или quantile transform;
* категориальные признаки: missing category + one-hot/hashing/frequency encoding;
* high-cardinality categorical: hashing trick или CatBoost native processing, но для SWDS нужен отдельный deterministic encoded representation;
* missing indicators добавлять явно;
* target encoding запрещён в основном SWDS-представлении, потому что может внести leakage через y.

Для drift score использовать только (X). Цель (y) нельзя использовать, кроме оценки downstream-деградации.

Для категориальных признаков лучше сделать два режима:

1. one-hot для low-cardinality;
2. hashing encoder для high-cardinality.

Обязательно зафиксировать размер hash-пространства, например 256 или 512. Иначе размерность будет гулять между датасетами.

## 10. Downstream-модели

Нельзя проверять drift score только на одной модели. Минимум три семейства:

1. **Logistic/Ridge/ElasticNet** — простой линейный baseline.
2. **LightGBM или XGBoost** — основной сильный классический ML baseline.
3. **CatBoost** — если много категориальных признаков.
4. Опционально: MLP/RealMLP-like small MLP как современный табличный DL sanity-check.

Основной claim лучше строить на LightGBM/CatBoost, потому что TabArena и TabReD показывают, что GBDT остаются сильными практическими моделями на табличных данных. ([arXiv][3])

Тюнинг:

* одинаковый небольшой budget для всех датасетов;
* random search на validation;
* 20 конфигураций максимум;
* early stopping;
* seed фиксировать;
* результаты усредлять по 3 seed, если это не слишком дорого.

Если времени мало: 1 seed для main, 3 seed для subset robustness.

## 11. Baseline drift scores

Нужно сравнить SWDS не с одним слабым baseline, а с группой.

Обязательные baseline:

1. **Mean KS**: среднее значение KS statistic по числовым признакам.
2. **Max KS**: максимум KS statistic по числовым признакам.
3. **Mean PSI**: средний population stability index.
4. **Max PSI**.
5. **MMD-RBF**: с median heuristic для bandwidth.
6. **Energy distance**.
7. **Classifier two-sample test**: обучить простой classifier различать reference window и current window; score = ROC-AUC различения.

Опционально:

8. ADWIN/KSWIN из river, если получится без боли.
9. PCA reconstruction drift.
10. Model confidence drift: изменение распределения predicted probabilities.

Важно: C2ST может быть сильным baseline, но он дороже и сам требует обучения. Поэтому если SWDS немного хуже C2ST, но сильно дешевле и стабильнее — это нормальный результат.

## 12. Основные метрики

Для drift detection:

* drift AUROC на synthetic drift;
* drift AUPRC;
* detection delay;
* false alarm rate;
* threshold stability.

Для связи с качеством модели:

* Spearman correlation между score и quality drop;
* Kendall tau как robustness;
* cross-correlation с лагами: score на окне (j) против падения качества на окне (j+1) или (j+2);
* regression ( \Delta Q_j \sim score_j ) с robust standard errors.

Для downstream performance:

Classification:

* ROC-AUC;
* PR-AUC;
* logloss;
* Brier score;
* ECE, если хватает времени.

Regression:

* RMSE;
* MAE;
* (R^2);
* pinball/quantile loss не нужен, если нет quantile-моделей.

Для retraining policy:

* cumulative regret;
* mean test quality;
* worst-window quality;
* number of retrains;
* quality per retrain;
* runtime overhead.

## 13. Статистические проверки

Без статистики работу легко разнести.

Основные проверки:

1. Для сравнения score-методов по dataset-model парам использовать **Wilcoxon signed-rank test**.
2. Для нескольких методов использовать **Friedman test** + post-hoc Nemenyi или pairwise Wilcoxon.
3. Для множественных сравнений применять **Benjamini–Hochberg FDR** или Holm.
4. Для корреляций давать **bootstrap 95% CI**.
5. Для runtime — median + IQR, не только mean.

Единица анализа: dataset-model pair, а не отдельное окно. Окна внутри одного датасета зависимы по времени, поэтому нельзя делать вид, что 100 окон — это 100 независимых экспериментов.

Это важная защита: рецензент может придраться к псевдорепликации. Надо заранее написать, что статистика считается по dataset-model pairs, а window-level результаты используются для построения scores и агрегатов.

## 14. Threshold calibration

Для deployment-сценария нужен threshold. Нельзя подбирать threshold на test.

Режим:

1. train model на train;
2. validation-period режется на окна;
3. на validation считается распределение SWDS при “нормальном” режиме;
4. threshold задаётся как quantile: 90%, 95%, 97.5%;
5. threshold фиксируется;
6. test-period используется только для оценки.

Для synthetic drift можно дополнительно строить ROC-кривые, потому что там есть ground truth drift/no-drift.

Основной deployment threshold: 95% quantile validation SWDS.

Robustness: показать sensitivity к 90/95/97.5%.

## 15. Window size ablation

Размер окна — типичный источник критики. Надо явно проверить:

* 500 объектов;
* 1000 объектов;
* 2000 объектов;
* 5000 объектов;

или адаптивно:

* 1%;
* 2.5%;
* 5%;
* 10% от train size.

Основной критерий: окно должно быть достаточно большим, чтобы метрики качества и SWDS не были слишком шумными.

Гипотеза: слишком маленькие окна дают много false alarms; слишком большие окна увеличивают detection delay. Оптимальный практический диапазон — 1000–5000 объектов, но это надо подтвердить эмпирически.

## 16. Ablation по числу проекций

Проверить:

[
K \in {16, 32, 64, 128, 256}
]

Зафиксировать random directions по seed и датасету, чтобы метод был воспроизводим.

Отчёт:

* SWDS-quality correlation vs K;
* runtime vs K;
* rank stability vs K;
* variance across seeds.

Рекомендованный основной режим: (K=128). Если хочется легче: (K=64), но тогда надо показать, что качество почти не падает.

## 17. Сценарии synthetic drift injection

Это отдельная обязательная часть.

Берёшь реальные (X), (y), но в test windows искусственно вносишь сдвиг в признаки. Нужно минимум 5 сценариев.

**Scenario A: mean shift**

Для случайной группы числовых признаков:

[
x'_d = x_d + \delta \cdot \sigma_d
]

где (\delta \in {0.25, 0.5, 1.0, 1.5}).

**Scenario B: variance shift**

[
x'_d = \mu_d + \gamma (x_d - \mu_d)
]

где (\gamma \in {1.25, 1.5, 2.0}).

**Scenario C: correlation shift**

Повернуть или смешать несколько числовых признаков линейной матрицей так, чтобы marginal distributions менялись слабее, чем joint distribution.

Это ключевой сценарий, где SWDS должен быть сильнее KS/PSI.

**Scenario D: categorical prior shift**

Для категориальных признаков изменить частоты категорий через resampling или category swap.

**Scenario E: missingness shift**

Увеличить вероятность missing values в группе признаков.

**Scenario F: local subpopulation shift**

Сдвиг затрагивает не весь поток, а 10–30% объектов, например только сегмент пользователей/магазинов/класса.

Это особенно важно, потому что реальные drift часто локальные.

## 18. Основной экспериментальный pipeline

Шаг 1. Скачать и привести датасеты к единому формату:

```text
dataset_name
X
y
timestamp
task_type
official_split if available
```

Шаг 2. Отсортировать по времени.

Шаг 3. Сделать train/val/test temporal split или использовать официальный split.

Шаг 4. Fit preprocessing только на train.

Шаг 5. Обучить downstream-модели на train, выбрать hyperparameters на val.

Шаг 6. Разрезать val и test на окна.

Шаг 7. Для каждого окна посчитать drift scores относительно reference.

Шаг 8. Для каждого окна посчитать downstream quality.

Шаг 9. Посчитать correlation между drift scores и quality drop.

Шаг 10. Провести synthetic drift injection и оценить detection metrics.

Шаг 11. Провести retraining-policy simulation.

Шаг 12. Провести ablations: K, window size, representation, threshold.

Шаг 13. Выполнить статистические тесты.

Шаг 14. Сформировать таблицы и графики.

## 19. Что должно быть в таблицах статьи

Минимальный набор таблиц:

**Table 1. Dataset summary**

Колонки:

* dataset;
* source;
* task;
* n_samples;
* n_features;
* n_numeric;
* n_categorical;
* missing rate;
* time span;
* number of test windows;
* metric.

**Table 2. Downstream model quality**

Колонки:

* dataset;
* model;
* validation quality;
* test initial-window quality;
* test final-window quality;
* quality drop.

**Table 3. Drift-quality correlation**

Строки: drift methods.
Колонки:

* median Spearman;
* mean rank;
* #wins;
* Wilcoxon p-value vs SWDS;
* adjusted p-value.

**Table 4. Synthetic drift detection**

Строки: drift methods.
Колонки:

* AUROC;
* AUPRC;
* detection delay;
* false alarm rate.

**Table 5. Retraining policy**

Строки: retraining policies.
Колонки:

* cumulative regret;
* mean quality;
* worst-window quality;
* number of retrains;
* runtime cost.

**Table 6. Ablation**

K, window size, representation.

## 20. Что должно быть на графиках

Обязательные графики:

1. Timeline plot: SWDS и model quality по времени на 2–3 датасетах.
2. Scatter plot: SWDS vs quality drop, с robust regression line.
3. Boxplot/rank plot: Spearman correlations по методам.
4. Detection delay plot на synthetic drift.
5. Runtime vs number of projections.
6. Retraining regret curves.
7. Ablation по window size.

Очень хорошо смотрится один figure типа “deployment story”: модель обучили, качество падает, SWDS растёт до падения, trigger срабатывает, retraining восстанавливает качество.

## 21. Acceptance criteria: когда работа считается успешной

Нужно заранее определить, что считается успехом, чтобы не получился cherry-picking.

Минимальный успех:

1. SWDS имеет медианную Spearman correlation с quality drop выше, чем mean KS, max KS, mean PSI, max PSI.
2. SWDS не хуже MMD/C2ST по correlation более чем на 5–10%, но быстрее или проще.
3. В synthetic covariance/correlation drift SWDS превосходит marginal baselines по AUROC или detection delay.
4. SWDS-triggered retraining имеет меньший regret, чем no retraining и не хуже periodic retraining при меньшем числе retrains.
5. Результаты сохраняются хотя бы на 6 датасетах.

Сильный успех:

1. SWDS входит в top-2 drift scores по большинству dataset-model pairs.
2. SWDS-triggered retraining превосходит periodic retraining при одинаковом бюджете.
3. (K=64) или (K=128) даёт стабильное качество и низкий runtime overhead.
4. Эффект сохраняется на classification и regression.

Если минимальный успех не выполнен, статью всё равно можно спасти как negative result:

**“When does Sliced Wasserstein fail as a drift proxy for tabular ML?”**

Но тогда нужно честно показать, что SWDS хорошо ловит covariate shift, но не concept drift без изменения X. Это нормальное ограничение, и оно даже методологически красиво.

## 22. Ограничения, которые надо признать заранее

Это обязательно нужно включить в статью, иначе рецензент сам это напишет.

1. SWDS видит сдвиг в (P(X)), но не обязан видеть чистый concept drift в (P(y|X)), если (P(X)) почти не меняется.
2. SWDS зависит от preprocessing mixed-type features.
3. High-cardinality categorical features могут искажать геометрию encoded space.
4. Drift score не равен performance drop: он только proxy.
5. Threshold переносится между датасетами плохо; threshold надо калибровать на validation stream.
6. Temporal windows зависимы, поэтому нельзя трактовать их как независимые наблюдения.
7. SWDS не заменяет полноценный мониторинг качества при наличии свежих labels.

Если эти ограничения прописать самому, это не ослабляет статью, а делает её взрослее.

## 23. Репозиторий и воспроизводимость

Структура репозитория:

```text
swds-tabular-drift/
  configs/
    datasets/
    models/
    experiments/
  data/
    raw/
    processed/
  src/
    data/
      loaders.py
      temporal_split.py
      preprocessing.py
    drift/
      sliced_wasserstein.py
      ks.py
      psi.py
      mmd.py
      energy.py
      c2st.py
    models/
      train.py
      evaluate.py
    monitoring/
      windows.py
      thresholds.py
      retraining_policy.py
    experiments/
      run_main.py
      run_synthetic_drift.py
      run_ablation.py
      run_retraining.py
    analysis/
      stats.py
      plots.py
  results/
    tables/
    figures/
  notebooks/
    01_sanity_checks.ipynb
    02_figures.ipynb
  README.md
  environment.yml
  pyproject.toml
```

Обязательные reproducibility-требования:

* все random seeds фиксируются;
* все конфиги сохраняются;
* все результаты пишутся в parquet/csv;
* каждый график строится из сохранённых таблиц, а не из “живого” notebook;
* preprocessing fitted only on train;
* test не используется для выбора threshold;
* датасеты документируются отдельной таблицей;
* для каждого исключённого датасета пишется причина исключения.

## 24. Минимальный стек

Python 3.11.

Библиотеки:

* numpy;
* pandas;
* scipy;
* scikit-learn;
* lightgbm;
* xgboost или catboost;
* matplotlib;
* seaborn можно для внутренней работы, но в статье графики лучше привести в едином стиле;
* pot необязательно, потому что 1D Wasserstein можно считать самому;
* river опционально для ADWIN/KSWIN.

GPU не нужен. Всё должно работать на CPU. Для ускорения SWDS можно использовать numpy vectorization.

## 25. Псевдокод SWDS

```python
def sliced_wasserstein_score(X_ref, X_cur, n_proj=128, n_quantiles=512, seed=42):
    rng = np.random.default_rng(seed)

    d = X_ref.shape[1]
    theta = rng.normal(size=(d, n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True) + 1e-12

    Z_ref = X_ref @ theta
    Z_cur = X_cur @ theta

    qs = np.linspace(0.0, 1.0, n_quantiles)

    Q_ref = np.quantile(Z_ref, qs, axis=0)
    Q_cur = np.quantile(Z_cur, qs, axis=0)

    w2_per_proj = np.mean((Q_ref - Q_cur) ** 2, axis=0)
    return float(np.sqrt(np.mean(w2_per_proj)))
```

Для sparse matrix нужен отдельный путь: `X @ theta` должен работать с scipy sparse. Если one-hot большой, не надо densify всю матрицу.

## 26. Как сформулировать claim в статье

Не надо писать:

“SWDS solves concept drift detection.”

Надо писать:

“We evaluate Sliced Wasserstein Drift Score as a lightweight, model-agnostic proxy for temporal covariate shift in tabular ML monitoring. We show that it is more informative than marginal drift indicators and can improve retraining decisions under limited labeling feedback.”

По-русски:

“Мы рассматриваем SWDS как лёгкий модельно-независимый индикатор временного ковариатного сдвига в табличных данных. Эксперименты показывают, что он лучше простых одномерных индикаторов связан с деградацией качества модели и может использоваться как практический триггер переобучения.”

## 27. План работ на 2 недели

День 1–2: loaders, preprocessing, temporal split, windowing.
День 3: downstream LightGBM/CatBoost training.
День 4: SWDS + KS/PSI baselines.
День 5: MMD/Energy/C2ST baselines.
День 6: main correlation experiment.
День 7: synthetic drift injection.
День 8: retraining policy simulation.
День 9: ablations по K/window/representation.
День 10: statistical tests.
День 11: figures/tables.
День 12: write methodology/results.
День 13: related work/limitations.
День 14: cleanup, README, reproducibility package.

## 28. Самая сильная версия итоговой статьи

Я бы в итоге сделал статью не как “мы придумали метрику”, а как:

**A Practical Evaluation of Sliced Wasserstein Drift Monitoring for Temporally Evolving Tabular Data**

Так звучит взрослее. В Q3/Q4 такое выглядит нормально: свежая проблема, строгий протокол, сильные baseline, минимум ресурсов, внятный прикладной вывод.

Главная мысль для защиты перед рецензентом:

**Мы не утверждаем, что SWDS универсально решает drift detection. Мы показываем, что в deployment-like temporal tabular setting он является дешёвым и более информативным proxy для мониторинга ковариатного сдвига, чем стандартные marginal indicators, особенно когда drift проявляется в совместной структуре признаков, а не в отдельных одномерных распределениях.**

[1]: https://arxiv.org/abs/2406.19380 "[2406.19380] TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks"
[2]: https://arxiv.org/abs/2504.12918 "[2504.12918] Sliced-Wasserstein Distance-based Data Selection"
[3]: https://arxiv.org/abs/2506.16791 "[2506.16791] TabArena: A Living Benchmark for Machine Learning on Tabular Data"

