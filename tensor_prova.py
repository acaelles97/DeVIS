import torch



if __name__ == "__main__":
    a = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    a_pair = a[:, 0::2][..., :-1]
    a_2_pair = a[:, 1::2]

    result = torch.stack((a[:, 0::2][..., :-1], a[:, 1::2]), dim=-1).flatten(1)

    cat_result = torch.cat((a[:, 0::2][..., :-1], a[:, 1::2]), dim=-1)
    to_change = cat_result[:, 1:-1].flip(-1)
    cat_result[:, 1:-1] = to_change