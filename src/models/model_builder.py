import os
import copy


import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from onmt.modules import Embeddings
from onmt.decoders.transformer import TransformerLMDecoder

from models.decoder import TransformerDecoder, LSTMDecoder
from models.encoder import TransformerEncoder
from models.optimizers import Optimizer
from models.tokenizer import Vocabulary
from models.vqvae.nearest_embed import NearestEmbed, EMAQuantize, EMAQuantizeList
from models.conv import Block
from models.encoder import PositionalEncoding

from others.logging import logger, init_logger

init_logger()

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None and not args.optim_init:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        logger.info("Initialize optimizer")
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

def init_module(net):
    for module in net.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def init_param(net):
    for p in net.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()


class ProteinLM(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(ProteinLM, self).__init__()
        self.args = args
        self.device = device

        self.vocab_size = args.vocab_size
        self.dec_hidden_size = self.args.dec_hidden_size

        # tgt_embeddings = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=0)
        tgt_embeddings = Embeddings(word_vocab_size=self.vocab_size, word_vec_size=self.dec_hidden_size, word_padding_idx=-1, position_encoding=True)

        self.decoder = TransformerLMDecoder(
            num_layers=self.args.dec_layers,
            d_model=self.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, copy_attn=True, self_attn_type='scaled-dot',
            dropout=self.args.dec_dropout, attention_dropout=self.args.dec_dropout,
            max_relative_positions=self.args.max_tgt_len,
            aan_useffn=True,
            embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = tgt_embeddings.word_lut.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print("Initialize the model!")
            init_module(self.decoder)
            init_param(self.generator)

        self.to(device)

    def forward(self, src, tgt, mask_src, mask_tgt):
        dec_input = tgt[:, :-1].permute(1, 0).unsqueeze(-1)     # [len, bz, nfeats=1]
        decoder_outputs, state = self.decoder(dec_input)
        decoder_outputs = decoder_outputs.permute(1, 0, 2)
        return decoder_outputs, None
        

class ProteinVAE(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(ProteinVAE, self).__init__()
        self.args = args
        self.device = device

        self.vocab_size = args.vocab_size
        self.enc_hidden_size = self.args.enc_hidden_size
        self.dec_hidden_size = self.args.dec_hidden_size
        self.latent_size = self.args.latent_size

        self.relu = nn.ReLU()
        self.mu_proj = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.logvar_proj = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.z_proj = nn.Linear(self.latent_size, self.dec_hidden_size)

        enc_embeddings = nn.Embedding(self.vocab_size, self.enc_hidden_size, padding_idx=0)
        tgt_embeddings = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=0)


        self.encoder = TransformerEncoder(
            num_layers=self.args.enc_layers,
            d_model=self.enc_hidden_size, heads=self.args.enc_heads,
            d_ff=self.args.enc_ff_size, dropout=self.args.enc_dropout, embeddings=enc_embeddings)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


    def forward(self, src, tgt, mask_src, mask_tgt):
        enc_output = self.encoder(src, mask_src)

        mu = self.mu_proj(enc_output)
        logvar = self.logvar_proj(enc_output)
        z = self.reparameterize(mu, logvar)     # [batch_size, seq_len, latent_dim]

        z = self.relu(self.z_proj(z))           # [batch_size, dec_hidden_size]
        dec_state = self.decoder.init_decoder_state(src)
        dec_output, _ = self.decoder(tgt[:, :-1], None, dec_state, condition=z[:,1:,:])

        return (dec_output, mu, logvar), None

    def reparameterize(self, mu, logvar):
        if self.args.mode == 'train':
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, batch_size):
        seed = torch.randn(batch_size, self.latent_size)
        trg = torch.zeros(batch_size, self.args.max_tgt_len, dtype=torch.long)
        trg[:, 0] = torch.ones(batch_size, dtype=torch.long)      # SENTENCE_START
        if self.cuda():
            seed = seed.cuda()
            trg = trg.cuda()
        seed = self.relu(self.z_proj(seed))

        state = self.decoder.init_decoder_state(batch_size, seed) # [batch_size]
        x = trg[:, 0].unsqueeze(-1)
        outputs = []
        for _ in range(trg.size(1)):
            output, state = self.decoder(x, state)
            scores = self.generator(output)
            outputs.append(scores.squeeze())
            x = scores.max(-1)[1]
        outputs = torch.stack(outputs, axis=1)

        return outputs, None

    def init_model(self, checkpoint):
        if checkpoint is not None:
            if self.args.train_from != '':
                self.load_pretrained(checkpoint['model'])
            else:
                self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print("Initialize the model!")
            init_module(self.decoder)
            if self.args.use_enc:
                init_module(self.encoder)
                if (self.args.share_emb) and (self.enc_hidden_size == self.dec_hidden_size):
                    # tgt_embeddings.weight = copy.deepcopy(enc_embeddings.weight)
                    self.encoder.embeddings.weight = self.decoder.embeddings.weight
            init_param(self.generator)

    def load_pretrained(self, checkpoint):
        model_dict = self.state_dict().copy()
        load_dict = {k:checkpoint[k] for k in model_dict.keys() if k in checkpoint.keys()}
        model_dict.update(load_dict)
        miss_key = [k for k in model_dict.keys() if k not in checkpoint.keys()]
        delete_key = [k for k in checkpoint.keys() if k not in model_dict.keys()]
        logger.info("Drop keys: {}".format(",".join(delete_key)))
        logger.info("Miss keys: {}".format(",".join(miss_key)))
        self.load_state_dict(model_dict, strict=True)


class ProteinVQVAE(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(ProteinVQVAE, self).__init__()
        self.args = args
        self.device = device

        self.vocab_size = args.vocab_size
        self.enc_hidden_size = self.args.enc_hidden_size
        self.dec_hidden_size = self.args.dec_hidden_size
        self.ss_size = self.args.ss_size
        self.latent_size = self.args.latent_size

        self.relu = nn.ReLU()
        self.latent_proj = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.z_proj = nn.Linear(self.latent_size, self.dec_hidden_size)
        self.vq_emb = NearestEmbed(num_embeddings=self.ss_size, embeddings_dim=self.args.latent_size)

        enc_embeddings = nn.Embedding(self.vocab_size, self.enc_hidden_size, padding_idx=0)
        tgt_embeddings = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=0)


        self.encoder = TransformerEncoder(
            num_layers=self.args.enc_layers,
            d_model=self.enc_hidden_size, heads=self.args.enc_heads,
            d_ff=self.args.enc_ff_size, dropout=self.args.enc_dropout, embeddings=enc_embeddings)

        self.decoder = LSTMDecoder(
            self.args.dec_layers,
            self.dec_hidden_size, heads=self.args.dec_heads,
            emb_dim=self.dec_hidden_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

    def forward(self, src, tgt, mask_src, mask_tgt):
        batch_size, seq_len = src.size(0), src.size(1)
        enc_output = self.encoder(src, mask_src)

        z_e = self.latent_proj(enc_output)
        z_e = z_e.view(-1, self.latent_size)

        z_q, argmin = self.vq_emb(z_e, weight_sg=True)
        z_emb, _ = self.vq_emb(z_e.detach())

        z_q = z_q.view(batch_size, seq_len, -1)
        z = self.relu(self.z_proj(z_q))

        top_vec = z.mean(axis=1)                # [batch_size, dec_hidden_size]
        init_state = self.decoder.init_decoder_state(batch_size, top_vec)
        dec_output, _ = self.decoder(tgt[:, :-1], init_state)
        z_e = z_e.view(batch_size, seq_len, -1)
        z_emb = z_emb.view(batch_size, seq_len, -1)

        return (dec_output, z_e, z_emb), argmin

    def sample(self, batch_size):
        seed = torch.randn(batch_size, self.latent_size)
        trg = torch.zeros(batch_size, self.args.max_tgt_len, dtype=torch.long)
        trg[:, 0] = torch.ones(batch_size, dtype=torch.long)      # SENTENCE_START
        if self.cuda():
            seed = seed.cuda()
            trg = trg.cuda()

        z_q, argmin = self.vq_emb(seed)
        z_q = self.relu(self.z_proj(z_q))

        state = self.decoder.init_decoder_state(batch_size, z_q) # [batch_size]
        x = trg[:, 0].unsqueeze(-1)

        scores, outputs = [], []
        for _ in range(trg.size(1)):
            output, state = self.decoder(x, state=state)
            score = self.generator(output)
            score = score.squeeze()
            scores.append(score)
            if self.args.sample == 'top':
                x = score.max(-1)[1]
                x = x.unsqueeze(1)
            elif self.args.sample == 'random':
                x = torch.multinomial(torch.exp(score), 1)
            else:
                raise NotImplementedError
            outputs.append(x)
        scores = torch.stack(scores, axis=1)
        outputs = torch.stack(outputs, axis=1).squeeze()

        return outputs, scores, None

    def init_model(self, checkpoint):
        if checkpoint is not None:
            if self.args.train_from != '':
                self.load_pretrained(checkpoint['model'])
            else:
                self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print("Initialize the model!")
            init_module(self.decoder)
            if self.args.use_enc:
                init_module(self.encoder)
                if (self.args.share_emb) and (self.enc_hidden_size == self.dec_hidden_size):
                    # tgt_embeddings.weight = copy.deepcopy(enc_embeddings.weight)
                    self.encoder.embeddings.weight = self.decoder.embeddings.weight
            init_param(self.generator)
            init_param(self.vq_emb)

    def load_pretrained(self, checkpoint):
        model_dict = self.state_dict().copy()
        load_dict = {k:checkpoint[k] for k in model_dict.keys() if k in checkpoint.keys()}
        model_dict.update(load_dict)
        miss_key = [k for k in model_dict.keys() if k not in checkpoint.keys()]
        delete_key = [k for k in checkpoint.keys() if k not in model_dict.keys()]
        logger.info("Drop keys: {}".format(",".join(delete_key)))
        logger.info("Miss keys: {}".format(",".join(miss_key)))
        self.load_state_dict(model_dict, strict=True)


class ProteinCVQVAE(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(ProteinCVQVAE, self).__init__()

        self.args = args
        self.device = device

        self.vocab_size = args.vocab_size
        self.enc_hidden_size = self.args.enc_hidden_size
        self.dec_hidden_size = self.args.dec_hidden_size
        self.ss_size = self.args.ss_size
        self.latent_size = self.args.latent_size
        self.k = self.args.entry_num * self.args.code_book

        self.relu = nn.ReLU()
        self.latent_proj = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.c_proj = nn.Linear(self.latent_size, self.dec_hidden_size)
        if args.sub_book > 1:
            self.vq_emb = EMAQuantizeList(args, dim=args.latent_size, n_embed=self.k, decay=args.decay, num=args.sub_book, step=1)
        else:
            self.vq_emb = EMAQuantize(dim=args.latent_size, n_embed=self.k, decay=args.decay)

        enc_embeddings = nn.Embedding(self.vocab_size, self.enc_hidden_size, padding_idx=0)
        tgt_embeddings = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=0)

        self.encoder = TransformerEncoder(
            num_layers=self.args.enc_layers,
            d_model=self.enc_hidden_size, heads=self.args.enc_heads,
            d_ff=self.args.enc_ff_size, dropout=self.args.enc_dropout, embeddings=enc_embeddings)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        # dilations = [1,2,4,8,16]
        dilations = [1,2,4,8,10]
        self.ss_input_layer = nn.Conv2d(1, self.args.conv_channel, kernel_size=(1, self.args.dec_hidden_size), padding='same')
        blocks = [Block(self.args.conv_channel, dilation=i) for i in dilations]
        self.ss_blocks = nn.Sequential(*blocks)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        self.ss_generator = get_generator(self.args.ss_size, self.args.latent_size, device)


    def forward(self, src, tgt, mask_src, mask_tgt):
        batch_size, seq_len = src.size(0), src.size(1)
        enc_output = self.encoder(src, mask_src)

        z_e = self.latent_proj(enc_output)
        # z_e = z_e.view(-1, self.latent_size)

        z_q, argmin, inputs, quantizes = self.vq_emb(z_e)

        z_q = z_q.view(batch_size, seq_len, -1)

        ss_condition = self.c_proj(z_q)
        dec_state = self.decoder.init_decoder_state(src)
        dec_output, _ = self.decoder(tgt[:, :-1], None, dec_state, condition=ss_condition[:,1:,:])

        if self.args.loss_func == 'cvqvae' or self.args.loss_func == 'crfvqvae':
            ss_input = self.ss_input_layer(z_q.unsqueeze(1))    # (batch_size, channel, seq_len, -1)
            ss_feats = self.ss_blocks(ss_input)
            ss_feats = torch.max(ss_feats, dim=1)[0]            # (batch_size, seq_len, -1)
        else:
            ss_feats = None

        inputs = inputs.view(batch_size, seq_len, -1)
        quantizes = quantizes.view(batch_size, seq_len, -1)
        argmin = argmin.view(batch_size, seq_len, -1)

        return (dec_output, inputs, quantizes, ss_feats), argmin

    def generate(self, src, tgt, mask_src, mask_tgt):
        batch_size, seq_len = src.size(0), src.size(1)
        enc_output = self.encoder(src, mask_src)

        z_e = self.latent_proj(enc_output)
        z_q, argmin, inputs, quantizes = self.vq_emb(z_e)
        z_q = z_q.view(batch_size, seq_len, -1)

        ss_condition = self.c_proj(z_q)
        dec_state = self.decoder.init_decoder_state(src)

        x = torch.ones(batch_size, 1, dtype=torch.long)
        if self.cuda():
            x = x.cuda()

        scores, outputs = [], []
        for t in range(seq_len-1):
            # print(x.size(), ss_condition[:,1:t+1,:].size())
            output, dec_state = self.decoder(x, None, dec_state, condition=ss_condition[:,1:t+2,:], step=t)
            score = self.generator(output)
            score = score[:, -1, :]
            scores.append(score)
            if self.args.sample == 'top':
                w = score.max(-1, keepdim=True)[1]
            elif self.args.sample == 'random':
                w = torch.multinomial(torch.exp(score), 1)
            else:
                raise NotImplementedError
            x = torch.cat([x, w], dim=-1)
            outputs.append(w)
        scores = torch.stack(scores, axis=1)
        outputs = torch.stack(outputs, axis=1).squeeze(-1)

        ss_input = self.ss_input_layer(z_q.unsqueeze(1))    # (batch_size, channel, seq_len, -1)
        ss_feats = self.ss_blocks(ss_input)
        ss_feats = torch.max(ss_feats, dim=1)[0]            # (batch_size, seq_len, -1)

        ss_scores = self.ss_generator(ss_feats)

        return outputs, scores, ss_scores

    def sample(self, embed_id, generation=True):
        """ embed_id [batch_size, seq_len, sub_book] """
        batch_size, seq_len = embed_id.size(0), embed_id.size(1)
        z_q  = self.vq_emb.embed_code(embed_id)
        z_q = z_q.view(batch_size, seq_len, -1)

        # ss
        ss_input = self.ss_input_layer(z_q.unsqueeze(1))    # (batch_size, channel, seq_len, -1)
        ss_feats = self.ss_blocks(ss_input)
        ss_feats = torch.max(ss_feats, dim=1)[0]            # (batch_size, seq_len, -1)
        ss_scores = self.ss_generator(ss_feats)

        if not generation:
            return None, None, ss_scores[:,1:]

        # generation init
        ss_condition = self.c_proj(z_q)
        dec_state = self.decoder.init_decoder_state(embed_id[:, :, 0])
        x = torch.ones(batch_size, 1, dtype=torch.long)
        x = x.cuda() if self.cuda() else x

        scores, outputs = [], []
        for t in range(seq_len-1):
            output, dec_state = self.decoder(x, None, dec_state, condition=ss_condition[:,1:t+2,:], step=t)
            score = self.generator(output)
            score = score[:, -1, :]
            scores.append(score)
            if self.args.sample == 'top':
                w = score.max(-1, keepdim=True)[1]
            elif self.args.sample == 'random':
                w = torch.multinomial(torch.exp(score), 1)
            else:
                raise NotImplementedError
            x = torch.cat([x, w], dim=-1)
            outputs.append(w)
        scores = torch.stack(scores, axis=1)
        outputs = torch.stack(outputs, axis=1).squeeze(-1)

        return outputs, scores, ss_scores[:,1:]
        # return outputs, scores, ss_scores

    def init_model(self, checkpoint):
        if checkpoint is not None:
            if self.args.train_from != '':
                self.load_pretrained(checkpoint['model'])
            else:
                self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print("Initialize the model!")
            init_module(self.decoder)
            if self.args.use_enc:
                init_module(self.encoder)
                if (self.args.share_emb) and (self.enc_hidden_size == self.dec_hidden_size):
                    # tgt_embeddings.weight = copy.deepcopy(enc_embeddings.weight)
                    self.encoder.embeddings.weight = self.decoder.embeddings.weight
            init_param(self.generator)
            init_param(self.ss_generator)

    def load_pretrained(self, checkpoint):
        model_dict = self.state_dict().copy()
        load_dict = {k:checkpoint[k] for k, v in model_dict.items() if k in checkpoint.keys() and v.size() == checkpoint[k].size()}
        model_dict.update(load_dict)
        miss_key = [k for k in model_dict.keys() if k not in load_dict.keys()]
        delete_key = [k for k in checkpoint.keys() if k not in load_dict.keys()]
        logger.info("Drop keys: {}".format(",".join(delete_key)))
        logger.info("Miss keys: {}".format(",".join(miss_key)))
        self.load_state_dict(model_dict, strict=True)
        # self.encoder.requires_grad_(False)
        # self.decoder.requires_grad_(False)


class ProteinCVQVAE2(ProteinCVQVAE):
    def __init__(self, args, device, checkpoint):
        super().__init__(args, device, checkpoint=checkpoint)

        self.enc_aa_embeddings = nn.Embedding(self.vocab_size, self.enc_hidden_size // 2, padding_idx=0)
        self.enc_ss_embeddings = nn.Embedding(self.ss_size, self.enc_hidden_size // 2, padding_idx=0)
        self.pos_embedding = PositionalEncoding(self.args.enc_dropout, self.enc_hidden_size // 2)

        self.encoder = TransformerEncoder(
            num_layers=self.args.enc_layers,
            d_model=self.enc_hidden_size, heads=self.args.enc_heads,
            d_ff=self.args.enc_ff_size, dropout=self.args.enc_dropout, embeddings=None)

    def forward(self, src, tgt, ss, mask_src, mask_tgt, prepand=True):
        batch_size, seq_len = src.size(0), src.size(1)

        if prepand:
            padding_ss_bos = torch.zeros((batch_size,1),dtype=torch.int)
            if self.cuda:
                padding_ss_bos = padding_ss_bos.cuda()
            padding_ss_eos = padding_ss_bos.clone()
            ss = torch.cat([padding_ss_bos, ss, padding_ss_eos], dim=-1)

        aa_emb = self.pos_embedding(self.enc_aa_embeddings(src))
        ss_emb = self.pos_embedding(self.enc_ss_embeddings(ss))
        enc_input = torch.cat([aa_emb, ss_emb], dim=-1)

        enc_output = self.encoder(enc_input, mask_src)

        z_e = self.latent_proj(enc_output)

        z_q, argmin, inputs, quantizes = self.vq_emb(z_e)

        z_q = z_q.view(batch_size, seq_len, -1)

        ss_condition = self.c_proj(z_q)
        dec_state = self.decoder.init_decoder_state(src)
        dec_output, _ = self.decoder(tgt[:, :-1], None, dec_state, condition=ss_condition[:,1:,:])

        if self.args.loss_func == 'cvqvae' or self.args.loss_func == 'crfvqvae':
            ss_input = self.ss_input_layer(z_q.unsqueeze(1))    # (batch_size, channel, seq_len, -1)
            ss_feats = self.ss_blocks(ss_input)
            ss_feats = torch.max(ss_feats, dim=1)[0]            # (batch_size, seq_len, -1)
        else:
            ss_feats = None

        inputs = inputs.view(batch_size, seq_len, -1)
        quantizes = quantizes.view(batch_size, seq_len, -1)
        argmin = argmin.view(batch_size, seq_len, -1)

        return (dec_output, inputs, quantizes, ss_feats), argmin

    def iterative_sample(self, embed_id, iteration=0):
        outputs, scores, ss_scores = self.sample(embed_id, generation=True)
        print(ss_scores.max(-1)[1])
        for _ in range(iteration):
            src = outputs
            ss = ss_scores.max(-1)[1]
            mask_src = ~(src == 0)
            _, embed_id = self.forward(src, src, ss, mask_src, mask_src, prepand=False)
            outputs, scores, ss_scores = self.sample(embed_id, generation=True)
            print(ss_scores.max(-1)[1])
        print()
        return outputs, scores, ss_scores
