import torch
import math
import torch.nn.functional as F
from mh.psi import psi


class MHs(torch.nn.Module):

    def __init__(self, q=1, d=2, m=3, u=4, delta=3,
                 gamma_alpha=0.1, tau_alpha=0.5, tau_mu=100, tau_eta=100,
                 upd_m=None, upd_k=None,
                 beta_k=0.001,
                 psi_fn="identity",
                 scramble=False,
                 random_memories=False,
                 key_input=False,
                 detach_gradients=True,
                 distance="cosine",
                 sigma_eu=1.0,
                 query_class_mappings=None,
                 memory_id_routing=False,
                 memory_id_mappings=None,
                 memory_id_size=None,
                 memory_id_type=None,
                 memory_id_propagation=None,
                 memory_id_proproute=None,
                 propagate=None):
        """
        :param q: number of neurons
        :param d: size of each key
        :param m: number of keys/memory units
        :param u: size of each memory unit
        :param gamma_alpha: softmax temperature (key matching)
        :param tau_alpha: threshold on the attention score of the winning key, to eventually trigger scrambling
        :param tau_mu: number of steps below which a key is considered to be not-used enough
        :param tau_eta: number of steps after which a key is considered old
        :param delta: number of top attention responses to select (top-delta)
        :param upd_m: update memory strategy (None, 'WTA')
        :param upd_k: update key strategy (None, 'ad_hoc_WTA', 'grad_WTA')
        :param beta_k: learning rate for key-update purposes when upd_k is 'ad_hoc_WTA'
        :param psi_fn: function to project the neuron input onto the key space
        :param scramble: triggers the key/memory scrambling routine when upd_k is 'ad_hoc_WTA'
        :param random_memories: start from random memories when scrambling
        :param key_input: use the network input for key-matching instead of the layer input
        :param detach_gradients: detach the flow of gradients trough layers
        :param distance: distance measure for input-key attention scores
        :param sigma_eu: temperature for euclidean distance
        :param query_class_mappings: contains the mappings for each class
        :param memory_id_routing: activate memory id routing mode
        :param memory_id_mappings: contains the mappings for each key/memory
        :param memory_id_type: top1 or topk memory routing
        :param memory_id_propagation: flag to avoid key size change when memory propagation is active
        :param memory_id_proproute: activate memory id routing propagation mode
        :param propagate: flag to concatenate memory id to layer output
        """

        super(MHs, self).__init__()
        assert upd_m in (None, 'WTA'), "Unknown value for upd_m, it must be None or 'WTA'"
        assert upd_k in (None, 'ad_hoc_WTA', 'grad_WTA'), "Unknown value for upd_k, it must be " \
                                                          "None, 'ad_hoc_WTA', or 'grad_WTA'"
        assert upd_m is None or (upd_m == 'WTA' and upd_k is not None), \
            "If upd_m is 'WTA', then upd_k must be ad_hoc_WTA or grad_WTA"
        assert distance in (
            "cosine", "euclidean", "dot_scaled"), "Unknown value for distance measure for input-key attention scores"
        self.q = q
        self.d = d
        self.m = m
        self.u = u
        self.gamma_alpha = gamma_alpha
        self.tau_alpha = tau_alpha
        self.tau_mu = tau_mu
        self.tau_eta = tau_eta
        self.scramble = scramble
        self.delta = min(delta, self.m)
        self.upd_m = upd_m
        self.upd_k = upd_k
        self.beta_k = beta_k
        self.psi_fn = psi_fn
        self.random_memories = random_memories
        self.key_input = key_input
        self.input_for_keys = None
        self.debug = False  # temporarily used
        self.detach_gradients = detach_gradients
        self.distance = distance
        self.sigma_eu = sigma_eu
        self.query_class_mappings = query_class_mappings
        self.memory_id_routing = memory_id_routing
        self.memory_id_mappings = memory_id_mappings
        self.memory_id_type = memory_id_type
        self.memory_id_propagation = memory_id_propagation
        self.memory_id_proproute = memory_id_proproute

        # creating keys (self.K) and memories (self.M)
        self.M = torch.nn.Parameter(torch.empty((self.q, self.m, self.u), dtype=torch.float32))
        if self.upd_k == "ad_hoc_WTA":
            self.register_buffer('K', torch.zeros((self.q, self.m, self.d)))
        else:
            self.K = torch.nn.Parameter(torch.empty((self.q, self.m, self.d), dtype=torch.float32))

        # buffers for key updates (average usefulness register buffer "mu" and age "eta")
        if self.scramble:
            self.register_buffer('mu', torch.zeros(self.q, m, dtype=torch.float))
            self.register_buffer('eta', torch.ones((self.q, m), dtype=torch.float) * self.tau_eta)
            # if self.debug:
            self.register_buffer('key_counter', torch.zeros(self.q, m, dtype=torch.float))
            # scrambling stats
            self.register_buffer('scrambling_count', torch.zeros(self.q, dtype=torch.long))
        else:
            self.mu = None
            self.eta = None

        # initializing memories and keys
        self.reset_parameters()

        self.x_bd_hook = None
        self.top_responses_bqk_hook = None
        self.top_indices_bqk_hook = None
        self.top_alpha_bqk_hook = None

        @torch.no_grad()
        def wrapper_hook(module, grad_input, grad_output):
            with torch.no_grad():
                    module.__update_keys_and_counters(self.x_bd_hook, self.top_responses_bqk_hook,
                                                      self.top_indices_bqk_hook,
                                                      self.top_alpha_bqk_hook)

        self.register_full_backward_hook(wrapper_hook)

    def reset_parameters(self):
        self.__reset_keys()
        self.__reset_memories()
        self.__reset_counters()

    def set_input_for_keys(self, x):
        self.input_for_keys = x

    def compute_weights(self, x, y=None):
        # shortcuts (notice that "self.delta" is called "k" in shortcuts, while "self.delta-1" is called "z")
        q, m, d, u, k, M_qmu = self.q, self.m, self.d, self.u, self.delta, self.M
        b = x.shape[0]

        # ensuring keys are normalized (not needed with ad_hoc_WTA updates)
        if self.distance == "cosine" and self.upd_k != "ad_hoc_WTA":
            self.__normalize_keys()
        else:
            if self.detach_gradients:
                x = x.detach()  # in ad-hoc WTA, no gradient is propagated to the layers below through key-matching

        # input used for key-matching is the one coming from the stream, for all the layers,
        # rather than from the previous layer
        if self.key_input:
            x = self.input_for_keys

        # mapping the input to the key space using the psi function
        x_bd = psi(x, self.psi_fn, key_size=d, normalize=self.distance == 'cosine',
                   query_class=y, query_class_mappings=self.query_class_mappings,
                   memory_routing=self.memory_id_routing, memory_id_mappings=self.memory_id_mappings,
                   input_for_keys=self.input_for_keys, routing_type=self.memory_id_type, training=self.training,
                   memory_id_propagation=self.memory_id_propagation)

        # finding the top responses and indices for the attention procedure
        top_responses_bqk, top_indices_bqk = self.__top_k_attention(x_bd)

        # probabilities
        top_alpha_bqk = torch.softmax(self.gamma_alpha * top_responses_bqk, dim=2)

        # if self.debug:
        # getting the top-1 indices for the current mini-batch
        if self.scramble:
            k_top1_indices_qb = top_indices_bqk[..., 0].t()
            self.key_counter.scatter_add_(dim=1,
                                          index=k_top1_indices_qb,
                                          src=torch.ones_like(k_top1_indices_qb, dtype=self.key_counter.dtype))

        # always set the top indices property
        self.top_indices_bqk_hook = top_indices_bqk
        # updating keys with the ad-hoc scheme (also refreshing top-stuff: responses, indices, alpha)

        if self.training:
            self.x_bd_hook = x_bd
            self.top_responses_bqk_hook = top_responses_bqk

            self.top_alpha_bqk_hook = top_alpha_bqk
            """
            top_responses_bqk, top_indices_bqk, top_alpha_bqk = \
                self.__update_keys_and_counters(x_bd, top_responses_bqk, top_indices_bqk, top_alpha_bqk)
            """
        # reading memories and blending them
        if self.upd_m is None:

            # preparing to read memory units and to blend them
            M_exp_bqmu = M_qmu.view(1, q, m, u).expand(b, q, m, u)

            # getting top memory units
            top_M_bqku = torch.gather(M_exp_bqmu, dim=2,
                                      index=top_indices_bqk.view(b, q, k, 1).expand(b, q, k, u))

            # mixing memory units by attention scores
            # -> top_alpha_bqk: [b,q,k], that we un-squeeze to [b,q,1,k]
            # -> top_M_bqku: [b,q,k,u]
            # -> W_bqu: matmul([(b,q),1,k], [(b,q),k,u]) = [b,q,1,u] that we squeeze to [b,q,u]
            W_bqu = torch.matmul(top_alpha_bqk.view(b, q, 1, k), top_M_bqku).squeeze(2)

        elif self.upd_m == 'WTA':

            # preparing to read memory units and to blend them
            M_exp_bqmu = M_qmu.view(1, q, m, u).expand(b, q, m, u)

            # dealing with top-1 stuff
            top1_M_exp_bq1u = torch.gather(M_exp_bqmu, dim=2,
                                           index=top_indices_bqk[..., 0:1].view(b, q, 1, 1).expand(b, q, 1, u))

            # mixing memory units by attention scores
            # -> top1_alpha_bqk: [b,q,k], that we select to [b,k,1] un-squeeze to [b,q,1,1]
            # -> top1_M_exp_bq1u: [b,q,1,u]
            # -> W_bqu: [b,q,1,1] * [b,q,1,u] = [b,q,1,u] that we squeeze to [b,q,u]
            top1_W_bqu = (top_alpha_bqk[..., 0:1].view(b, q, 1, 1) * top1_M_exp_bq1u).squeeze(2)

            # dealing with top-2-and-following stuff
            top2on_M_exp_bqzu = torch.gather(M_exp_bqmu.detach(), dim=2,
                                             index=top_indices_bqk[..., 1:].view(b, q, k - 1, 1).expand(b, q, k - 1, u))
            top2on_alpha_bqz = top_alpha_bqk[:, :, 1:]
            if self.upd_k == 'grad_WTA':
                top2on_alpha_bqz = top2on_alpha_bqz.detach()

            # mixing memory units by attention scores
            # -> top2on_alpha_bqz: [b,q,k-1], that we un-squeeze to [b,q,1,k-1]
            # -> top2on_M_exp_bqzu: [b,q,k-1,u]
            # -> W_bqu: matmul([(b,q),1,k-1], [(b,q),k-1,u]) = [b,q,1,u] that we squeeze to [b,q,u]
            top2on_W_bqu = torch.matmul(top2on_alpha_bqz.view(b, q, 1, k - 1), top2on_M_exp_bqzu).squeeze(2)

            # merging top1 and top-2-and-following stuff
            W_bqu = top1_W_bqu + top2on_W_bqu

        else:

            # what is going on?
            raise NotImplementedError

        return W_bqu

    def forward(self, x, y=None):
        raise NotImplementedError

    @torch.no_grad()
    def __create_random_memories(self, n):
        M_new = torch.empty((self.q, n, self.u), dtype=self.M.dtype, device=self.M.device)
        bound = 1. / math.sqrt(self.u)
        torch.nn.init.uniform_(M_new, -bound, bound)
        return M_new

    @torch.no_grad()
    def __reset_memories(self):
        bound = 1. / math.sqrt(self.u)
        torch.nn.init.uniform_(self.M, -bound, bound)

    @torch.no_grad()
    def __reset_keys(self):
        bound = 1. / math.sqrt(self.d)
        torch.nn.init.uniform_(self.K, -bound, bound)
        if self.distance == 'cosine':
            self.K.data = F.normalize(self.K, p=2.0, dim=2, eps=1e-12, out=None)

    def __reset_counters(self):
        if self.mu is not None:
            self.mu = torch.zeros_like(self.mu)
        if self.eta is not None:
            self.eta.fill_(self.tau_eta)

    def reset_key_counter(self):
        self.key_counter = torch.zeros_like(self.key_counter)

    def __top_k_attention(self, x_bd):

        if self.distance != 'euclidean':
            # matmul([b,d], [d,qm]) = [b,qm], then reshaped (view) to [b,q,m]
            responses_bqm = torch.matmul(x_bd, self.K.view(self.q * self.m, self.d).t()).view(-1, self.q,
                                                                                              self.m) / math.sqrt(
                self.d)
        else:
            b, d, m, q = x_bd.shape[0], self.d, self.m, self.q
            x_b1d = x_bd.view(b, 1, d)
            K_b_qm_d = self.K.view(1, self.q * self.m, self.d).expand(b, self.q * self.m, self.d)
            dists_bqm = torch.cdist(x_b1d, K_b_qm_d, p=2).pow(2).view(b, self.q, self.m)
            responses_bqm = torch.exp(-dists_bqm / (math.sqrt(d) * self.sigma_eu))

        top_responses_bqk, top_indices_bqk = torch.topk(responses_bqm, k=self.delta, dim=2, largest=True, sorted=True)
        return top_responses_bqk, top_indices_bqk

    @torch.no_grad()
    def __normalize_keys(self, ids=None):
        """
        :param ids: None or a vector with "self.q" elements, with the indices of the keys to consider for each neuron
        """
        if ids is not None:
            ids_exp_q1d = ids.view(self.q, 1, 1).expand(self.q, 1, self.d)
            keys_q1d = torch.gather(self.K, dim=1, index=ids_exp_q1d)
            keys_q1d = F.normalize(keys_q1d, p=2.0, dim=2, eps=1e-12, out=None)
            self.K.scatter_(dim=1, index=ids_exp_q1d, src=keys_q1d)
            return keys_q1d
        else:
            self.K.data = F.normalize(self.K, p=2.0, dim=2, eps=1e-12, out=None)
            return self.K

    def __update_keys_and_counters(self, x_bd, top_responses_bqk, top_indices_bqk, top_alphas_bqk):

        # saving some shortcuts, notice that "self.delta" is called "k" here
        b, q, d, m, k, u = x_bd.shape[0], self.q, self.d, self.m, self.delta, self.u
        K_qmd = self.K
        mu_qm, eta_qm = self.mu, self.eta

        # getting the top-1 indices for the current mini-batch
        top1_indices_qb = top_indices_bqk[..., 0].t()

        # temporarily used
        if self.debug:
            K_initial_qmd = self.K.clone()
            M_initial_qmd = self.M.clone()
            mu_initial_qm = self.mu.clone()
            eta_initial_qm = self.eta.clone()
            top1_indices_initial_qb = top1_indices_qb.clone()
        else:
            K_initial_qmd, M_initial_qmd, mu_initial_qm = None, None, None
            eta_initial_qm, top1_indices_initial_qb = None, None

        # determining if we need to scramble keys and memories (if scrambling is enabled)
        # (up to one key/memory per neuron, even when the batch size is greater than one)
        if self.scramble:

            # computing a boolean mask that tells what neurons should be subject to scrambling (scramble_q),
            # a boolean mask associated with the weak keys,
            # and the indices of the elements of x_bd (batch elements) that should replace the
            # scrambled keys (weak_batch_elements_q)
            scramble_q, weak_keys_mask_qm, weak_batch_elements_q = \
                self.__evaluate_scrambling_conditions(top_responses_bqk)

            # temporarily used
            if self.debug:
                self.__debug_pre_scrambling(top_indices_bqk, top_alphas_bqk, scramble_q,
                                            weak_keys_mask_qm, weak_batch_elements_q, x_bd)

            # if at least one neuron requires a scrambling operation, we do scramble! (in-place)
            if torch.any(scramble_q):
                self.__scramble(x_bd, scramble_q, weak_keys_mask_qm, weak_batch_elements_q, top1_indices_qb)

            # temporarily used
            if self.debug:
                self.__debug_changes_with_respect_to(K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                                     top1_indices_initial_qb, top1_indices_qb,
                                                     msg="Right after scrambling...")

        # manually update keys only in the case of ad_hoc_WTA
        if self.upd_k == 'ad_hoc_WTA':
            # computing variations to apply to winning keys, eventually using an adaptive learning rate
            if self.distance == 'cosine':
                # delta to add to each winning key # TODO se cosine no
                key_variations_qbd = (self.beta_k * x_bd).view(1, b, d).expand(q, b, d)
            elif self.distance == 'dot_scaled':
                # delta to add to each winning key
                key_variations_qbd = ((self.beta_k / math.sqrt(d)) * x_bd).view(1, b, d).expand(q, b,
                                                                                                d)
            elif self.distance == 'euclidean':
                keys_qbd = K_qmd.gather(dim=1, index=top1_indices_qb.view(q, b, 1).expand(q, b, d))
                key_variations_qbd = \
                    self.beta_k / (
                            math.sqrt(d) * self.sigma_eu) * top_responses_bqk[:, :, 0].T.view(q, b, 1) * (
                            x_bd.view(1, b, d).expand(q, b, d) - keys_qbd)  # delta to add to each winning key
            else:
                raise NotImplementedError
            # TODO here the issue for detach
            # updating the winning keys (one winning key per neuron, for each batch example)
            K_qmd.scatter_add_(dim=1,
                               index=top1_indices_qb.view(q, b, 1).expand(q, b, d),
                               src=key_variations_qbd)  # adding variations to winning keys

            # recomputing or updating responses (in the case of batch size > 1, we recompute them all)
            top_responses_bqk, top_indices_bqk, top1_indices_qb = \
                self.__update_top_k_attention(x_bd, top_responses_bqk, top_indices_bqk, top1_indices_qb)

            # recomputing the softmax over the (now updated) top responses
            top_alphas_bqk = torch.softmax(self.gamma_alpha * top_responses_bqk, dim=2)  # [b,q,k]

        if self.scramble:
            # resetting ages of winning keys
            eta_qm.scatter_(1, top1_indices_qb, 0.)

            # updating counters: usages ("mu") for the winning keys and ages ("eta") for all the keys
            mu_qm.scatter_add_(dim=1,
                               index=top1_indices_qb,
                               src=torch.ones_like(top1_indices_qb, dtype=mu_qm.dtype))  # winning keys
            eta_qm += b  # all the keys

        # temporarily used
        if self.debug:
            self.__debug_changes_with_respect_to(K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                                 top1_indices_initial_qb, top1_indices_qb,
                                                 msg="At the end of the whole key-and-counters update procedure...")

        return top_responses_bqk, top_indices_bqk, top_alphas_bqk

    def __evaluate_scrambling_conditions(self, top_responses_bqk):

        # shortcuts
        mu_qm = self.mu
        eta_qm = self.eta

        # computing the max of alphas (we take the smallest of them in case of batch sizes greater than one)
        top1_responses_bq = top_responses_bqk[..., 0]  # max of responses
        max_of_responses_q, weak_batch_elements_q = torch.min(top1_responses_bq, dim=0)

        # finding the weak keys, if any (boolean mask)
        weak_keys_mask_qm = torch.logical_and(mu_qm < self.tau_mu, eta_qm >= self.tau_eta)

        # determining on what neurons scrambling should be applied (boolean mask)
        scramble_q = torch.logical_and(max_of_responses_q < self.tau_alpha, torch.any(weak_keys_mask_qm, dim=1))

        return scramble_q, weak_keys_mask_qm, weak_batch_elements_q

    def __scramble(self, x_bd, scramble_q, weak_keys_mask_qm, weak_batch_elements_q, top1_indices_qb):

        # shortcuts
        q, d, u = self.q, self.d, self.u
        K_qmd, M_qmu, mu_qm, eta_qm = self.K, self.M, self.mu, self.eta

        # stats
        self.scrambling_count[scramble_q] += 1

        # finding the indices of the candidate keys to scramble (one per neuron)
        scramble_candidates_keys_q = torch.max(eta_qm * weak_keys_mask_qm.to(torch.float), dim=1)[1]

        # neurons that must and must-not be subject to scrambling operations
        scramble_q = scramble_q.to(torch.float)
        no_scramble_q = torch.logical_not(scramble_q).to(torch.float)  # boolean mask

        # scrambling keys
        scramble_candidates_keys_exp_q1d = scramble_candidates_keys_q.view(q, 1, 1).expand(q, 1, d)
        scramble_q11 = scramble_q.view(q, 1, 1)
        no_scramble_q11 = no_scramble_q.view(q, 1, 1)

        new_keys_q1d = x_bd.gather(dim=0, index=weak_batch_elements_q.view(q, 1).expand(q, d)).view(q, 1, d)
        old_keys_q1d = K_qmd.gather(dim=1, index=scramble_candidates_keys_exp_q1d)

        self.K.data = torch.scatter(K_qmd, dim=1,
                                    index=scramble_candidates_keys_exp_q1d,
                                    src=new_keys_q1d * scramble_q11 + old_keys_q1d * no_scramble_q11)
        """
        K_qmd.scatter_(dim=1,
                       index=scramble_candidates_keys_exp_q1d,
                       src=new_keys_q1d * scramble_q11 + old_keys_q1d * no_scramble_q11)
        """

        # scrambling memories

        with torch.no_grad():
            scramble_candidates_keys_exp_q1u = scramble_candidates_keys_q.view(q, 1, 1).expand(q, 1, u)
            weak_keys_q1 = top1_indices_qb.gather(dim=1, index=weak_batch_elements_q.view(q, 1))
            if self.random_memories:
                new_memories_q1u = self.__create_random_memories(1)
            else:
                weak_keys_exp_q1u = weak_keys_q1.view(q, 1, 1).expand(q, 1, u)
                new_memories_q1u = M_qmu.gather(dim=1, index=weak_keys_exp_q1u)
            old_memories_q1u = M_qmu.gather(dim=1, index=scramble_candidates_keys_exp_q1u)
            """
            M_qmu.scatter_(
                dim=1,
                index=scramble_candidates_keys_exp_q1u,
                src=new_memories_q1u * scramble_q11 + old_memories_q1u * no_scramble_q11)
            """
            self.M.data = torch.scatter(M_qmu, dim=1,
                                        index=scramble_candidates_keys_exp_q1u,
                                        src=new_memories_q1u * scramble_q11 + old_memories_q1u * no_scramble_q11)

        # updating indices of the winning keys (in-place!)
        scramble_q1 = scramble_q.view(q, 1)
        no_scramble_q1 = no_scramble_q.view(q, 1)

        """        
        weak_batch_elements_q1 = weak_batch_elements_q.view(q, 1)
        new_top1_indices_q1 = scramble_candidates_keys_q.view(q, 1)
        old_top1_indices_q1 = top1_indices_qb.gather(dim=1, index=weak_batch_elements_q1)
        top1_indices_qb.scatter_(
            dim=1, index=weak_batch_elements_q1,
            src=(new_top1_indices_q1 * scramble_q1 + old_top1_indices_q1 * no_scramble_q1).to(torch.long))
        """
        # resetting to zero the usage counts ("mu") for keys that were scrambled
        scramble_candidates_keys_q1 = scramble_candidates_keys_q.view(q, 1)

        new_values = 0.
        old_values_q1 = mu_qm.gather(dim=1, index=scramble_candidates_keys_q1)

        mu_qm.scatter_(
            dim=1,
            index=scramble_candidates_keys_q1,
            src=new_values * scramble_q1 + old_values_q1 * no_scramble_q1)

    def __update_top_k_attention(self, x_bd, top_responses_bqk, top_indices_bqk, top1_indices_qb):
        q, d, b = self.q, self.d, x_bd.shape[0]

        if b == 1:
            # normalizing the winning keys (recall that b=1 here)
            if self.distance == "cosine":
                norm_winning_keys_q1d = self.__normalize_keys(ids=top1_indices_qb)  # recall that b=1 here
                # updating responses (the response with the updated key is recomputed, for each neuron)
                response_winning_bq1 = torch.matmul(x_bd, norm_winning_keys_q1d.view(q, d).t()).view(b, q,
                                                                                                     1)
            elif self.distance == "dot_scaled":
                ids_exp_q1d = top1_indices_qb.view(self.q, 1, 1).expand(self.q, 1, self.d)
                norm_winning_keys_q1d = torch.gather(self.K, dim=1, index=ids_exp_q1d)
                # updating responses (the response with the updated key is recomputed, for each neuron)
                response_winning_bq1 = torch.matmul(x_bd, norm_winning_keys_q1d.view(q, d).t()).view(b, q,
                                                                                                     1) / math.sqrt(
                    d)
            elif self.distance == "euclidean":

                # reshaping stuff (recall that b=1 here)
                ids_exp_q1d = top1_indices_qb.view(self.q, 1, 1).expand(self.q, 1, self.d)
                keys_q1d = torch.gather(self.K, dim=1, index=ids_exp_q1d)
                x_b1d = x_bd.view(b, 1, d)
                K_bqd = keys_q1d.view(1, self.q, self.d)
                dists_bq1 = torch.cdist(x_b1d, K_bqd, p=2).pow(2).view(b, self.q, 1)

                # updating responses (the response with the updated key is recomputed, for each neuron)
                response_winning_bq1 = torch.exp(-dists_bq1 / (math.sqrt(d) * self.sigma_eu))
            else:
                raise NotImplementedError

            top_responses_bqk[:, :, 0] = response_winning_bq1.squeeze()

        elif b > 1:

            # normalizing all the keys (for simplicity - with large batch sizes it is likely easier/faster)
            if self.distance == 'cosine':
                self.__normalize_keys()

            # recomputing all responses, re-determining the top-responses
            top_responses_bqk, top_indices_bqk = self.__top_k_attention(x_bd)

            # re-transposing top-1 indices
            top1_indices_qb = top_indices_bqk[..., 0].t()

        return top_responses_bqk, top_indices_bqk, top1_indices_qb

    def __str__(self):
        s = "- q = " + str(self.q) + "\n"
        s += "- d = " + str(self.d) + "\n"
        s += "- m = " + str(self.m) + "\n"
        s += "- u = " + str(self.u) + "\n"
        s += "- delta = " + str(self.delta) + "\n"
        s += "- gamma_alpha = " + str(self.gamma_alpha) + "\n"
        s += "- tau_alpha = " + str(self.tau_alpha) + "\n"
        s += "- tau_mu = " + str(self.tau_mu) + "\n"
        s += "- tau_eta = " + str(self.tau_eta) + "\n"
        s += "- upd_m = " + str(self.upd_m) + "\n"
        s += "- upd_k = " + str(self.upd_k) + "\n"
        s += "- beta_k = " + str(self.beta_k) + "\n"
        s += "- psi_fn = " + self.psi_fn + "\n"
        s += "- scramble = " + str(self.scramble)
        return s

    def __debug_pre_scrambling(self, top_indices_bqk, top_alphas_bqk, scramble_q,
                               weak_keys_mask_qm, weak_batch_elements_q, x_bd):
        b, q, k, d, m = top_indices_bqk.shape[0], self.q, self.delta, self.d, self.m
        mu_qm, eta_qm = self.mu, self.eta

        # finding the indices of the candidate keys to scramble (one per neuron)
        scramble_candidates_keys_q = torch.max(eta_qm * weak_keys_mask_qm.to(torch.float), dim=1)[1]

        print("*** __debug_pre_scrambling ***")
        print("No scrambling is applied at all...not yet...")
        s = ""
        torch.set_printoptions(profile='full', linewidth=2000)
        for i in range(0, b):
            if i > 0:
                s += "\n"
            for j in range(0, q):
                if j > 0:
                    s += "\n"
                s += "[batch element " + str(i)
                s += ", neuron " + str(j) + "] " if q > 1 else "]"
                s += "\n- tau_alpha, tau_mu, tau_eta: " + str(self.tau_alpha) + ", " \
                     + str(self.tau_mu) + ", " + str(self.tau_eta)
                s += "\n- mu:       " + str(mu_qm[j, :])
                s += "\n- eta:      " + str(eta_qm[j, :])
                s += "\n- top-keys (alphas): "
                for a in range(0, k):
                    s += str(top_indices_bqk[i, j, a].item())
                    s += " ({0:.3g})".format(top_alphas_bqk[i, j, a].item())
                    if a < k - 1:
                        s += ", "
                s += "\n- scramble? " + str(scramble_q[j].item())
                if scramble_q[j].item() is True:
                    s += "\n  - what key? " + str(scramble_candidates_keys_q[j].item())
                    s += "\n  - with what batch element? " + str(weak_batch_elements_q[j].item())
                    s += "\n  - data of such element? "
                    s += str(x_bd[weak_batch_elements_q[j], 0:min(3, d)])
                    if d > 3:
                        s += " ~printing only the first 3 components"
        print(s)
        torch.set_printoptions(profile='default')

    def __debug_changes_with_respect_to(self, K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                        top1_indices_initial_qb, top1_indices_qb, msg=None):
        q, m, b, u, d = self.q, self.m, top1_indices_initial_qb.shape[1], self.u, self.d

        changed_keys_qm = torch.greater(torch.max(torch.abs(self.K - K_initial_qmd), dim=2)[0], 1e-5)
        changed_memories_qm = torch.greater(torch.max(torch.abs(self.M - M_initial_qmd), dim=2)[0], 0.)
        changed_top1_indices_qb = torch.greater(torch.abs(top1_indices_initial_qb.to(torch.float) -
                                                          top1_indices_qb.to(torch.float)), 0.)
        changed_mus_qm = torch.greater(torch.abs(self.mu - mu_initial_qm), 0.)
        changed_etas_qm = torch.greater(torch.abs(self.eta - eta_initial_qm), 0.)

        print("*** __debug_changes_with_respect_to ***")
        if msg is not None:
            print(msg)
        s = ""
        for j in range(0, q):
            if j > 0:
                s += "\n"
            s += "[neuron " + str(j) + "]"
            num_changed_keys = torch.sum(changed_keys_qm[j, :]).item()
            num_changed_memories = torch.sum(changed_memories_qm[j, :]).item()
            num_changed_top1_indices = torch.sum(changed_top1_indices_qb[j, :]).item()
            num_changed_mus = torch.sum(changed_mus_qm[j, :]).item()
            num_changed_etas = torch.sum(changed_etas_qm[j, :]).item()
            s += "\n- #changed keys: " + str(num_changed_keys)
            if num_changed_keys > 0:
                if d > 3:
                    s += " ~printing only the first 3 components"
                for t in range(0, m):
                    if changed_keys_qm[j, t].item() is True:
                        s += "\n  - key " + str(t) + ": " + str(self.K[j, t, 0:min(3, d)])
                        s += " (it was: " + str(K_initial_qmd[j, t, 0:min(3, d)]) + ")"
            s += "\n- #changed memory units: " + str(num_changed_memories)
            if num_changed_memories > 0:
                if u > 3:
                    s += " ~printing only the first 3 components"
                for t in range(0, m):
                    if changed_memories_qm[j, t].item() is True:
                        s += "\n  - mem " + str(t) + ": " + str(self.M[j, t, 0:min(3, u)])
                        s += " (it was: " + str(M_initial_qmd[j, t, 0:min(3, u)]) + ")"
            s += "\n- #changed top1 indices: " + str(num_changed_top1_indices)
            if num_changed_top1_indices > 0:
                for i in range(0, b):
                    if changed_top1_indices_qb[j, i].item() is True:
                        s += "\n  - batch element " + str(i) + ": " + str(top1_indices_qb[j, i].item())
                        s += " (it was: " + str(top1_indices_initial_qb[j, i].item()) + ")"
            s += "\n- #changed mus: " + str(num_changed_mus)
            if num_changed_mus > 0:
                for t in range(0, m):
                    if changed_mus_qm[j, t].item() is True:
                        s += "\n  - mu " + str(t) + ": " + str(self.mu[j, t].item())
                        s += " (it was: " + str(mu_initial_qm[j, t].item()) + ")"
            s += "\n- #changed etas: " + str(num_changed_etas)
            if num_changed_etas > 0:
                for t in range(0, m):
                    if changed_etas_qm[j, t].item() is True:
                        s += "\n  - eta " + str(t) + ": " + str(self.eta[j, t].item())
                        s += " (it was: " + str(eta_initial_qm[j, t].item()) + ")"
        print(s)
