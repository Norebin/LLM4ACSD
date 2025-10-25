import torch
import torch.utils.data as data_utils
import tqdm

from logic.csvreader import read_functions
from logic.embeds.utils import save_to_file
from transformers import AutoTokenizer, AutoModel


def build_tokens(code_path):
    tokenizer = AutoTokenizer.from_pretrained("/model/LiangXJ/Model/PLM/bert-base-nli-mean-tokens")
    lines = read_functions(code_path)
    encoded_input = tokenizer(lines, padding=True, truncation=True, return_tensors='pt', max_length=510)
    return encoded_input


def build(code_path, file_name, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained("/model/LiangXJ/Model/PLM/bert-base-nli-mean-tokens")
    model.to(device)

    token_tensor = build_tokens(code_path)

    keys = list(token_tensor.keys())
    ds = data_utils.TensorDataset(*[i.to(device) for i in list(token_tensor.values())])
    loader = data_utils.DataLoader(ds, batch_size=batch_size)

    iterator = tqdm.tqdm(enumerate(loader), total=len(loader))

    embeds = []
    for i, j in iterator:
        with torch.no_grad():
            temp_dict = {keys[i]:j[i] for i in range(3)}
            output = model(**temp_dict)
            embeds.append(output)
    print("Shape of embeddings is", embeds[0][0].shape if batch_size != 1 else embeds[0].shape)

    if file_name is None:
        file_name = "bert_nli_pooler"

    save_to_file(embeds, file_name, "/model/lxj/Baseline/TripletLoss/data")
