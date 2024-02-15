{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMl7Pu5zMe9yp/RI30/gBYL"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer\n",
        "import signal\n",
        "import sys\n",
        "from PIL import Image\n",
        "import torch\n",
        "import os\n",
        "import json\n",
        "from collections import defaultdict\n",
        "\n",
        "def exit_gracefully(signum, frame):\n",
        "    signal.signal(signal.SIGINT, original_sigint)\n",
        "    sys.exit(1)\n",
        "\n",
        "model = VisionEncoderDecoderModel.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
        "processor = ViTImageProcessor.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "\n",
        "def saveTextFile(text):\n",
        "    try:\n",
        "        print(text)\n",
        "        with open(\"output_data.txt\", \"w+\") as text_file:\n",
        "            text_file.write(text)\n",
        "    except Exception as e:\n",
        "        print(\"Exception occurred\\n\")\n",
        "        print(e)\n",
        "\n",
        "def read_image():\n",
        "    path_to_file = '/content/img1.jpg'\n",
        "    image = Image.open(path_to_file)\n",
        "    if image.mode != \"RGB\":\n",
        "        image = image.convert(mode=\"RGB\")\n",
        "    return [image]\n",
        "\n",
        "max_length = 16\n",
        "num_beams = 4\n",
        "gen_kwargs = {\"max_length\": max_length, \"num_beams\": num_beams}\n",
        "\n",
        "def predict_step(images):\n",
        "    pixel_values = processor(images=images, return_tensors=\"pt\").pixel_values\n",
        "    pixel_values = pixel_values.to(device)\n",
        "\n",
        "    output_ids = model.generate(pixel_values, **gen_kwargs)\n",
        "\n",
        "    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)\n",
        "    preds = [pred.strip() for pred in preds]\n",
        "    return preds\n",
        "\n",
        "def tag_from_data(preds):\n",
        "    awsstring = \"I think it is \" + str(preds[0])\n",
        "    return awsstring\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    original_sigint = signal.getsignal(signal.SIGINT)\n",
        "    signal.signal(signal.SIGINT, exit_gracefully)\n",
        "    images = read_image()\n",
        "    data = predict_step(images)\n",
        "    text = tag_from_data(data)\n",
        "    saveTextFile(text)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DXykI8dhMNPD",
        "outputId": "a7039dbf-50bd-4532-f2ac-033b1dd7b58d"
      },
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "I think it is a woman and a man are looking at a cell phone\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "OwebHX-pNha2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}