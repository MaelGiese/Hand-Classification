# Hand-Classification

## Utilitaire de prise de photo
- Lancer le script : python create_classification_image.py
- Choisir une pose de main parmis les poses proposées (Fist, OK, Palm).
- Une fois que la fenêtre de webcam est lancée un carré vert devrait apparaître autour d’une main si il y en a une dans l’image, cela signifie que le programme détecte une main.
 - Appuyer sur ‘s’ pour enregistrer cette image de main.
 - Si aucune main n’est détectée, aucune image ne sera enregistrée en appuyant sur ‘s’.Le carré vert devient rouge si une image est enregistrée.
 - Pour fermer la fenêtre appuyer sur 

## Stockage des photos
Les photos sont stockées dans le fichier “hand_classification/Dataset” à partir de la racine du projet.

Arborescence des dossiers de stockage
[!(https://raw.githubusercontent.com/MaelGiese/Hand-Classification/master/Arbo.png)]
Les données sont réparties en données de test et d'entraînement pour pouvoir comparer l’amélioration du modèle de classification de manière fiable (le but est d’avoir toujours les mêmes données de test).
On a un dossier pour chaque classe (Fist, OK, Palm), chacun de ces dossiers est réparti en jeu de données appelé ‘SET_X’.
Chaque jeu de données est créé par l’utilitaire de photo a chaque lancement.

Les photos prises par l’utilitaire sont enregistrées au format .png, la taille de chaque image n’est pas modifiée à l’enregistrement, elle varie donc pour chaque image.

## Données augmentées
L’augmentation de données semble être adaptée à notre problème de classification de poses de main étant donnée du manque de différent fond dans nos images d'entraînement.

Traitement effectué au images :
- rotation horizontale et verticale
- saturation
- modification de la luminosité
- zoom
- image grisé

Toutes les modification effectuées aux images se trouve dans le fichier hand_classification.utils.data_augmentation

## Entraînement
Les données d'entraînement du modèle on étaient augmentées, on a 3 000 images d'entraînement en ajoutant les données augmenté on arrive à 32 000 images et 780 images de test.

### Hyperpamètres :
- batch_size = 512
- epochs = 65
- learning_rate = 0.001


### Résultat 
[!(https://raw.githubusercontent.com/MaelGiese/Hand-Classification/master/Training.png)]
- Train loss: 0.1501
- Train accuracy: 0.9457

- Test loss: 0.40203176679758307
- Test accuracy: 0.8923077


## Demo :
[![](http://img.youtube.com/vi/8GA2EqDS1TM/0.jpg)](http://www.youtube.com/watch?v=8GA2EqDS1TM "Demo")

## Notice :
https://docs.google.com/document/d/1lE9REC806BY9a9yu7aqw8YeNyGQ-wasRhewcrX4AqMY/edit?usp=sharing
