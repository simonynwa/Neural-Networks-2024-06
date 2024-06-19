# Classification des genres musicaux

Ce code fournit des réseaux de neurones pour classer le genre musical. 

## Utilisation
Pour utiliser le code, exécuter le code:

```bash
python main.py
```

### Dépendances Python
```
numpy
```

## Brève introduction

#### Data préparation
Nous choississons d’utiliser l’ensemble de données GTZAN et sélectionné cinq de ses formes musicales：blues, classical, jazz, pop et metal.

En utilisant ``audioreader.py``, nous avons extrait les fonctionnalités audio et les avons écrites dans le fichier CSV ``audio_features.csv``, puis nous les avons divisées en un ensemble d’entraînement et un ensemble de test selon le principe 80/20, et nous les avons écrites dans des fichiers CSV respectivement ``test_audio_features.csv`` et ``train_audio_features.csv``.

#### Models basés sur Numpy

Nous n’avons utilisé numpy que pour construire la couche convolutive, la couche entièrement connectée, la fonction d’activation, la couche de pooling, etc., et avons implémenté la propagation avant et réciproque dans chaque couche. Et vous pouvez les voir dans le classeur ``models``.

#### Construire des réseaux de neurones
Nous avons combiné ces couches dans notre modèle de réseau de neurones et comparé les résultats. Vous pouvez voir le fichier ``model.py`` dans ``models``.

#### Optimisation des hyperparamètres
Afin d’obtenir les meilleurs résultats du modèle, nous avons utilisé une méthode de recherche aléatoire pour déterminer le taux d’apprentissage et la taille du minibatch. Vous pouvez exécuter le code:
```bash
python hyperparameter_search.py
```

#### Comparaison avec les méthodes du machine learning

Nous avons comparé des méthodes d’apprentissage automatique telles que KNN, Kmeans, forêt aléatoire, Support Vector Machine (SVM), etc., et avons conclu que le modèle de réseau neuronal fonctionne mieux. Vous pouvez exécuter le code:
```bash
python KNN.py
python Kmeans.py
python randomforest.py
python SVC.py
```


