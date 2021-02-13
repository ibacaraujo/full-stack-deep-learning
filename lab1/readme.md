# Lab 1. Estrutura básica do projeto para reconhecimento no conjunto de dados MNIST

Para entender a estrutura geral do projeto vamos treinar uma MLP usando os dados do MNIST.

Vamos cobrir:

- Estrutura do projeto
- Um modelo simples em PyTorch
- Treinamento baseado no PyTorch-Lightning
- Um jeito único de executar os experimentos: `python3 run_experiment.py`

## Estrutura do projeto

Nesse primeiro laboratório, foi explicado que o código vai ser construído incrementalmente.
Haverá um laboratório por semana mostrando mais código.

Para esse primeiro laboratório, conhecemos a estrutura do projeto:

```sh
(fsdl-text-recognizer-2021) ➜  lab1 git:(main) ✗ tree -I "logs|admin|wandb|__pycache__"
.
├── readme.md
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   └── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── util.py
└── training
    ├── __init__.py
    └── run_experiment.py
```

O código está organizado em dois pontos principais, `text_recognizer` and `training`.

O `text_recognizer` pode ser visto como um pacote Python que vamos desenvolver e eventualmente colocar em produção.

O `training` pode ser visto como código de suporte para o pacote `text_recognizer`. Atualmente, além do `__init__.py` comum para módulos do Python, consiste apenas de `run_experiment.py`.

Dentro do `text_recognizer`, o código está organizado em `data`, `models`, and `lit_models`.

Vamos aprender sobre cada um deles.

### Data

Há três escopo para lidarmos com os dados com nomes semelhantes: `DataModule`, `DataLoader`, and `Dataset`.

No nível de cima temos a classe `DataModule` que é responsável por algumas coisas:

- Baixar os dados brutos e/ou gerar dados sintéticos
- PyTorch models Processar os dados necessários para serem usados pelos modelos em PyTorch
- Dividir os dados em treino, validação e teste
- Especificar as dimensões das entradas em termos do número de canais, tamanho da altura e tamanho da largura das imagens
- Especificar informação sobre as classes
- Especificar transformações de aumento de dados para usar no treinamento

Para cumprir essas funcionalidades, `DataModule` faz uso das seguintes classes:

1. `torch Dataset`, que retorna instâncias individuais e opcionalmente transformadas dos dados.
2. `torch DataLoader`, que amostra batches, embaralha a ordem das imagens nesses batches e entrega essas imagens para a GPU.

Para ler mais sobre essas [interfaces de dados do PyTorch](https://pytorch.org/docs/stable/data.html).

Para evitar ficar escrevendo código repetido para todos os dados, definimos uma classe simples `text_recognizer.data.BaseDataModule` que herda de [`pl.LightningDataModule`](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html).
Essa herança vai nos permitir usar os dados de forma muito simples com PyTorch-Lightning `Trainer` e evitar problemas com treinamento distribuído.

### Models

Modelos são as redes neurais. São simplesmente código que aceitam uma entrada, processa-a a partir de camadas e produz uma saída.

O código é parcialmente escrito e parcialmente aprendido. Ele é escrito em relação à arquitetura das redes neurais e é aprendido em relação aos parâmetros de todas as camadas da arquitetura.
Portanto, a computação do modelo deve ser retropropagável.

Como estamos usando PyTorch, todos os modelos são subclasses de `torch.nn.Module`, que torna os torna aprendíveis dessa maneira. 

### Lit Models

Usamos PyTorch-Lightning para treinamento, que define a interface `LightningModule` para lidar não apenas com tudo o que um Modelo precisa, mas também para especificar os detalhes do algoritmo de aprendizado: qual a função de perda deve ser utilizada na saída do modelo e o ground truth, qual otimizador deve ser usado e com que taxa de aprendizado, etc.

## Training

Agora entendemos o suficiente para treinar.

O `training/run_experiment.py` é um script que lida com muitos parâmetros de linha de comando.

Abaixo um comando que podemos executar:

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1
```

Enquanto `model_class` e `data_class` são nossos próprios argumentos, `max_epochs` e `gpus` são argumentos automaticamente selecionados do `pytorch_lightning.Trainer`.
Você pode usar qualquer outra flag do `Trainer` (ver [docs](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags)) na linha de comando, por exemplo `--batch_size=512`. 

O script `run_experiment.py` também seleciona flags das classes do modelo e dos dados que são especificados.
Por exemplo, em `text_recognizer/models/mlp.py` nós especificamos a classe `MLP` e adicionamos algumas flags de linha de comando, `--fc1` e `--fc2`.

De acordo com isso, podemos executar:

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1 --fc1=4 --fc2=8
```

E assistir o modelo falhar em atingir uma acurácia alta devido a poucos parâmetros.

## Homework

- Tentar `training/run_experiment.py` com diferentes hiperparâmetros do MLP (por exemplo, `--fc1=128 --fc2=64`).
- Tentar editar a arquitetura de MLP em `text_recognizers/models/mlp.py`
- Explicar o que você fez e colar a saída do seu treinamento.
