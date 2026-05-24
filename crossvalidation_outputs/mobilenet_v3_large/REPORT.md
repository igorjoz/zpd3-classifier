# MobileNetV3-Large - raport cross-validation

## Konfiguracja eksperymentu

- Liczba foldów: `5`
- Epoki: `12`
- Batch size: `64`
- Warmup epochs: `3`
- Unfreeze blocks: `3`
- Learning rate classifier: `0.001`
- Learning rate feature: `0.0001`
- Weight decay: `0.0001`

## Wyniki foldów

| Fold | Validation macro F1 | Validation loss | Best epoch |
| ---: | ---: | ---: | ---: |
| 1 | 0.7458 | 1.2004 | 12 |
| 2 | 0.8517 | 0.6812 | 11 |
| 3 | 0.8588 | 0.4920 | 10 |
| 4 | 0.8650 | 0.4046 | 10 |
| 5 | 0.7812 | 0.6137 | 2 |

## Agregacja wyników

- Średnie validation macro F1: `0.8205`
- Odchylenie standardowe validation macro F1: `0.0480`
