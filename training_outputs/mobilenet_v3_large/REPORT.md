# MobileNetV3-Large - raport baseline

## Preparation

Problem zostal potraktowany jako wieloklasowa klasyfikacja obrazow (7 klas naczyn). Wybrana metoda to transfer learning z modelem MobileNetV3-Large wytrenowanym wstepnie na ImageNet. Ostatnia warstwa klasyfikatora zostala zastapiona warstwa wyjsciowa dla 7 klas.

Model jest lekki obliczeniowo, dlatego stanowi rozsadny pierwszy baseline dla zbioru o ograniczonej liczbie obrazow oraz uruchomienia bez wykrytej karty NVIDIA.

## Metody oceny

1. **Loss (CrossEntropyLoss)** - mierzy blad optymalizowany w treningu; obserwujemy train i validation loss po kazdej epoce.
2. **Accuracy** - udzial poprawnych predykcji; latwy do interpretacji przy wzglednie wyrownanych klasach.
3. **Macro F1-score** - srednia F1 liczona rowno dla wszystkich klas; jest glownym kryterium wyboru checkpointow.
4. **ROC-AUC One-vs-Rest macro** - mierzy zdolnosc rankingu prawdopodobienstw oddzielnie dla kazdej klasy.
5. **Confusion matrix** - pokazuje konkretne pary mylonych klas i pozwala diagnozowac bledy modelu.
6. **Balanced accuracy** - dodatkowa kontrola, w ktorej kazda klasa wnosi rowny wklad.

## Podzial danych

Podzial wykonano grupowo po `object_id`, aby ujecia tego samego fizycznego obiektu nie znalazly sie jednoczesnie w treningu i ewaluacji.

| Split | Obrazy | Udzial |
| --- | ---: | ---: |
| train | 1275 | 70.02% |
| validation | 273 | 14.99% |
| test | 273 | 14.99% |

Kontrola przecieku grup: **zaliczona**.

## Hiperparametry

- Epoki: `12`
- Batch size: `64`
- Optymalizator: `AdamW`, weight decay `0.0001`
- Learning rate glowicy: `0.001`
- Learning rate fine-tuningu cech: `0.0001`
- Etap zamrozonego ekstraktora: `3` epoki
- Odblokowane koncowe bloki features: `3`
- Augmentacja tylko dla train: losowy crop, odbicie poziome, mala rotacja i zmiana kolorow.
- Normalizacja: srednie i odchylenia ImageNet zgodne z wagami pretrained.
- Cache CPU: obraz jest jednorazowo skalowany do krotszego boku `256` px przed transformacjami epoki.

## Wyniki walidacji i wybor modelu

Checkpointy `MODELS_BASELINE` wybrano wylacznie na podstawie validation macro F1 (przy remisie nizszy validation loss). Zbior testowy nie bral udzialu w wyborze.

| Ranking | Epoka | Validation macro F1 | Validation loss | Etap |
| ---: | ---: | ---: | ---: | --- |
| 1 | 9 | 0.7816 | 0.7473 | fine_tune |
| 2 | 12 | 0.7762 | 0.8201 | fine_tune |
| 3 | 7 | 0.7647 | 0.8122 | fine_tune |

Najlepszy checkpoint pochodzi z epoki `9`: validation accuracy `0.7949`, macro F1 `0.7816`, ROC-AUC `0.9660`.

## Jednorazowa ocena testowa

Po wybraniu najlepszego checkpointu na walidacji wykonano jedna ocene na odlozonym zbiorze testowym.

| Loss | Accuracy | Macro F1 | Balanced accuracy | ROC-AUC macro OvR |
| ---: | ---: | ---: | ---: | ---: |
| 0.4184 | 0.8645 | 0.8629 | 0.8625 | 0.9850 |

## Analiza wyboru

- Roznica train-validation macro F1 w najlepszej epoce wynosi `0.2109`, co wskazuje na przeuczenie. Nastepny eksperyment powinien zwiekszyc regularyzacje lub skrocic fine-tuning.
- Najlepszy wynik pojawil sie przed koncem treningu; zapis checkpointow ochronil baseline przed pogorszeniem pozniejszych epok.
- Macro F1 na tescie jest wyzsze od walidacji o `0.0813`. Nie sluzy to do ponownego wyboru modelu, ale wskazuje na rozna trudnosc grup w splitach; kolejnym potwierdzeniem powinna byc grupowa walidacja krzyzowa.

## Artefakty

- `history.csv` i `training_curves.png`: przebieg metryk po kazdej epoce.
- `per_epoch/*_confusion_matrix.png`: macierze pomylek train i validation dla kazdej epoki.
- `per_epoch/*_roc.png`: krzywe ROC train i validation dla kazdej epoki.
- `selected_validation_*` oraz `test_*`: diagnostyka wybranego baseline na walidacji i koncowym tescie.
- `MODELS_BASELINE/*.pt`: trzy najlepsze modele na podstawie walidacji.
