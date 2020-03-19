# SemEval 2018 Task 1 - Emotion Recogntion Tweets
This project aims at solving the emotion recognition task from [SemEval 2018, Task 1](https://competitions.codalab.org/competitions/17751).

## Data pre-processing
The following pre-processing steps were performed on the dataset:
- Transforming encoded emoticons into words (😀 -> grinning face)
- Lemmatizing words (went -> go)
- Replacing named entites and tweeter usernames with tokens (The White House -> GPE (geopolitical entity))
- Removing stop words, uncessary punctuation and stripping off hashtags.

## Models
We tried two different models: an SVM classifier using the radial basis function as its kernel and a random forest with 100 estimators.

## Results

### Performance Metrics

| Model        | Accuracy           | Micro F1  | Macro F1 |
| :-------------: |:-------------:| :-----:| :-----:|
|   SVM    | 0.449 | 0.58 | 0.45 |
| Random Forest      | 0.427     |  0.56  | 0.40 |

### Confusion Matrices - Random Forest
Anger
| Positive | Negative |
|:---------:|:-------:|
| 1917 | 241 |
| 474 | 627 |

Anticipation
| Positive | Negative |
|:---------:|:-------:|
| 2802 | 32 |
| 414 | 11 |

Disgust
| Positive | Negative |
|:---------:|:-------:|
| 1805 | 355 |
| 518 | 581 |

Fear
| Positive | Negative |
|:---------:|:-------:|
| 2689 | 85 |
| 213 | 272 |

Joy
| Positive | Negative |
|:---------:|:-------:|
| 1524 | 293 |
| 423 | 1019 |

Love
| Positive | Negative |
|:---------:|:-------:|
| 2631 | 112 |
| 338 | 178 |

Optimism
| Positive | Negative |
|:---------:|:-------:|
| 1823 | 293 |
| 575 | 568 |

Pessimism
| Positive | Negative |
|:---------:|:-------:|
| 2836 | 48 |
| 333 | 42 |

Sadness
| Positive | Negative |
|:---------:|:-------:|
| 2145 | 154 |
| 571 | 389 |

Surprise
| Positive | Negative |
|:---------:|:-------:|
| 3069 | 20 |
| 161 | 9 |

Trust
| Positive | Negative |
|:---------:|:-------:|
| 3096 | 10 |
| 151 | 2 |

### Confusion Matrices - SVM

Anger
| Positive | Negative |
|:---------:|:-------:|
| 1885 | 273 |
| 439 | 662 |

Anticipation
| Positive | Negative |
|:---------:|:-------:|
| 2582 | 252 |
| 336 | 89 |

Disgust
| Positive | Negative |
|:---------:|:-------:|
| 1750 | 410 |
| 430 | 669 |

Fear
| Positive | Negative |
|:---------:|:-------:|
| 2676 | 98 |
| 242 | 243 |

Joy
| Positive | Negative |
|:---------:|:-------:|
| 1582 | 235 |
| 443 | 999 |

Love
| Positive | Negative |
|:---------:|:-------:|
| 2573 | 170 |
| 279 | 237 |

Optimism
| Positive | Negative |
|:---------:|:-------:|
| 1754 | 362 |
| 465 | 678 |

Pessimism
| Positive | Negative |
|:---------:|:-------:|
| 2712 | 172 |
| 280 | 95 |

Sadness
| Positive | Negative |
|:---------:|:-------:|
| 2051 | 248 |
| 483 | 477 |

Surprise
| Positive | Negative |
|:---------:|:-------:|
| 3048 | 41 |
| 160 | 10 |

Trust
| Positive | Negative |
|:---------:|:-------:|
| 3067 | 39 |
| 149 | 4 |
