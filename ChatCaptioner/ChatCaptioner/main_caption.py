import os
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from chatcaptioner.chat import set_openai_key, caption_images, get_instructions
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import PosterDataset, plot_img, print_info


def parse():
    parser = argparse.ArgumentParser(
        description="Generating captions in test datasets."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="path to the dataset csv file"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="output/",
        help="root path for saving results",
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        required=True,
        help="tag for this experiment. caption results will be saved in save_root/exp_tag",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10,
        help="Number of QA rounds between GPT3 and BLIP-2. Default is 10, which costs about 2k tokens in GPT3 API.",
    )
    parser.add_argument(
        "--n_blip2_context",
        type=int,
        default=1,
        help="Number of QA rounds visible to BLIP-2. Default is 1, which means BLIP-2 only remember one previous question. -1 means BLIP-2 can see all the QA rounds",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt",
        choices=[
            "gpt3",
            "chatgpt",
            "text-davinci-003",
            "text-davinci-002",
            "davinci",
            "gpt-3.5-turbo-1106",
            "FlanT5XXL",
            "OPT",
        ],
        help="model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system",
    )
    parser.add_argument("--device_id", type=int, default=0, help="Which GPU to use.")
    parser.add_argument(
        "--genre_sub",
        type=bool,
        default=False,
        help="Whether to use an alternative genre",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="../../poster",
        help="Path to the image folder",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Set OpenAI
    openai_key = os.environ["OPENAI_API_KEY"]
    set_openai_key(openai_key)

    # Load BLIP-2
    blip2s = {
        "FlanT5 XXL": Blip2(
            "FlanT5 XXL", device_id=args.device_id, bit8=True
        ),  # load BLIP-2 FlanT5 XXL to GPU0. Too large, need 8 bit. About 20GB GPU Memory
    }

    if args.model == "FlanT5XXL":
        question_model = blip2s["FlanT5 XXL"]
    elif args.model == "OPT":
        question_model = Blip2("OPT6.7B", device_id=2, bit8=True)
    else:
        question_model = args.model

    # load the dataset
    dataset = PosterDataset(
        args.dataset, genre_sub=args.genre_sub, image_path=args.image_path
    )

    # preparing the folder to save results
    save_path = os.path.join(args.save_root, args.exp_tag)
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "caption_result"))
    with open(os.path.join(save_path, "instruction.yaml"), "w") as f:
        yaml.dump(get_instructions(), f)

    # start caption
    caption_images(
        blip2s,
        dataset,
        save_path=save_path,
        n_rounds=args.n_rounds,
        n_blip2_context=args.n_blip2_context,
        model=question_model,
        print_mode="bar",
    )


if __name__ == "__main__":
    args = parse()
    main(args)
