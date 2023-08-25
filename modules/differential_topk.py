import torch
from torch import nn
import torch.nn.functional as F

###########################################
############# differential topK ###########
###########################################
# Calculation of differential topK is based on [Top-K](https://arxiv.org/pdf/2104.03059.pdf), thanks
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int=500, sigma: float=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int=500, sigma: float=0.05):
        # input here is scores with (bs, num_patches)
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(dtype=x.dtype, device=x.device)
        perturbed_x = x.unsqueeze(1) + noise*sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = F.one_hot(indices, num_classes=d).float() # b, nS, k, d
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None]*5)
        
        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None]*5)

###########################################
############# differential topK ###########
###########################################

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
            # nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :]
        global_x = x[:,:1, :]
        # print("global_x.shape: ", global_x.shape)
        x = torch.cat([local_x, global_x.expand(B, N, C)], dim=-1)
        return self.out_conv(x)

class VisualTokenSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
    
    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D) # shape here is (bs, max_frames, n_patches, hid_dim)(128, 12, 50, 512)
        x = x.reshape(-1, N, D) # shape here is (bs*max_frames, n_patches, hid_dim)
        pred_score = self.score_predictor(x).squeeze() # (bs*max_frames, n_patches)
        
        spatial_pred_score = pred_score[:, 1:] # seperate the cls_token (bs*max_frames, n_patches-1)
        topk_indicator = self.topk_selector(spatial_pred_score) # (bs*max_frames, k, n_patches-1))

        # cls token as cls token
        cls_x_feature = x[:, :1, :] # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :] # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, spatial_x_feature)

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1) # shape here is (bs*max_frames, topkPatches, hid_dim)(128, 12, 4, 512)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D) # shape here is (B, max_frames*topkPatches, D) 

        return output

class STPredictorConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            #  nn.Tanh()
            nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, max_frames):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B_frame, N, C = x.size()
        B = B_frame // max_frames
        local_x = x[:,:, :]

        global_x = x[:,:1, :].reshape(B, max_frames, 1, C) # shape (bs, n_length, cls_tokens, hid_dim)
        global_x = torch.mean(global_x, 1, True).expand(B, max_frames, 1, C).reshape(B_frame, 1, C)
        # print("global_x.shape: ", global_x.shape)

        x = torch.cat([local_x, global_x.expand(B_frame, N, C)], dim=-1)
        return self.out_conv(x)


class STVisualTokenSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = STPredictorConv(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
    
    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D) # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D) # shape here is (bs*max_frames, n_patches, hid_dim)
        pred_score = self.score_predictor(x, self.max_frames).squeeze() # (bs*max_frames, n_patches)
        
        spatial_pred_score = pred_score[:, 1:] # seperate the cls_token (bs*max_frames, n_patches-1)
        topk_indicator = self.topk_selector(spatial_pred_score) # (bs*max_frames, k, n_patches-1))

        # cls token as cls token
        cls_x_feature = x[:, :1, :] # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :] # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, spatial_x_feature)

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1) # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D) # shape here is (B, max_frames*topkPatches, D) 

        return output

class VisualTokenRandomSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.topk = topk
    
    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D) # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D) # shape here is (bs*max_frames, n_patches, hid_dim)

        # cls token as cls token
        cls_x_feature = x[:, :1, :] # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :] # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        patch_len = spatial_x_feature.shape[1]
        selected_indices = torch.randperm(patch_len)[:self.topk].sort()[0]
        selected_patch_feature = spatial_x_feature[:, selected_indices, :]

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1) # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D) # shape here is (B, max_frames*topkPatches, D) 

        return output

class TextPredictorLG(nn.Module):
    """ Text to Patch Embedding
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x, text):
        '''
        x: shape (bs, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :]
        global_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].unsqueeze(1)
        x = torch.cat([local_x, global_x.expand(B, N, C)], dim=-1)
        return self.out_conv(x)

class TextTokenSelection(nn.Module):
    def __init__(self, embed_dim=512, topk=1):
        super().__init__()
        self.score_predictor = TextPredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
    
    def forward(self, x, input_ids, attention_mask, training=True):
        '''
        x: input embed, shape is (bs, max_words, hid_dim)
        input_ids: (bs, max_words) token id, cls is the max
        attention_mask: (bs, max_words)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''      
        B, N, D = x.shape
        pred_score = self.score_predictor(x, input_ids).squeeze() # (bs, max_words)
        
        attention_mask_new = torch.cat((attention_mask[:, 1:], torch.zeros(B,1).to(device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
        # print("attention_mask: ", attention_mask[0], "\nattention_mask_new: ", attention_mask_new[0])
        word_pred_score = pred_score*attention_mask_new # seperate the cls_token (bs, n_token-1)
        # print("word_pred_score: ", word_pred_score[0])
        topk_indicator = self.topk_selector(word_pred_score) # (bs, k, n_token-1))

        # cls token as cls token
        cls_x_feature = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)].unsqueeze(1) # cls_token, shape here is (bs, 1, hid_dim)

        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, x)

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1) # shape here is (bs, topkPatches, hid_dim)

        return output
