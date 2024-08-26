from collections import defaultdict
from typing import List

import dgl
import sherpa.algorithms
import torch
from loguru import logger

from shared.enums import FeatureEnum
from shared.user import User


def get_user_item_map(train: List[User]):
    user_map = {u.index: i + 1 for i, u in enumerate(train)}
    items = {i for user in train for i, _ in user.ratings}
    item_map = {item: idx + 1 for idx, item in enumerate(sorted(items))}

    return user_map, item_map


def construct_user_item_heterogeneous_graph(meta, g, self_loop=False, norm='both', global_nodes=False):
        users = torch.LongTensor(meta.users)
        items = torch.LongTensor(meta.items)
        n_entities = len(meta.entities)

        user_ntype = 'user'
        item_ntype = 'item'

        # Get user-item edges while changing user indices.
        u, i = g.edges()
        u = u - n_entities
        ui = u, i
        iu = i, u

        if not self_loop:
            graph_data = {
                (user_ntype, 'ui', item_ntype): ui,
                (item_ntype, 'iu', user_ntype): iu
            }
        else:
            graph_data = {
                (user_ntype, 'ui', item_ntype): ui,
                (item_ntype, 'iu', user_ntype): iu,
                (user_ntype, 'self_user', user_ntype): (users, users),
                (item_ntype, 'self_item', item_ntype): (items, items)
            }

        n_users, n_items = len(meta.users), len(meta.items)
        if global_nodes:
            graph_data[(user_ntype, 'global_user', user_ntype)] = (torch.full_like(users, len(users)), users)
            graph_data[(item_ntype, 'global_item', item_ntype)] = (torch.full_like(items, len(items)), items)
            n_users += 1
            n_items += 1

        new_g = dgl.heterograph(graph_data, num_nodes_dict={'user': n_users, 'item': n_items})
        new_g.nodes['user'].data['recommendable'] = torch.zeros(new_g.num_nodes('user'), dtype=torch.bool)
        new_g.nodes['item'].data['recommendable'] = torch.ones(new_g.num_nodes('item'), dtype=torch.bool)

        if norm == 'both':
            for etype in new_g.etypes:
                # get degrees
                src, dst = new_g.edges(etype=etype)
                dst_degree = new_g.in_degrees(dst, etype=etype).float()  # obtain degrees
                src_degree = new_g.out_degrees(src, etype=etype).float()

                # calculate norm in eq. 3 of both ngcf and lgcn papers.
                norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)  # compute norm
                new_g.edges[etype].data['norm'] = norm

        return new_g

def deterministic_feature_getter(features, nodes, meta):
    input_features = {}
    mask = torch.zeros_like(nodes)
    for enum in FeatureEnum:
        if enum in features:
            if enum == FeatureEnum.USERS:
                selection = nodes >= len(meta.entities)
            elif enum == FeatureEnum.ITEMS:
                selection = nodes < len(meta.items)
            elif enum == FeatureEnum.DESC_ENTITIES:
                selection = torch.logical_and(nodes < len(meta.entities), nodes >= len(meta.items))
            elif enum == FeatureEnum.ENTITIES:
                selection = nodes < len(meta.entities)
            else:
                raise NotImplementedError()

            # Ensure features have not been used previously. I.e. items are a subgroup of entities.
            selection = torch.logical_and(torch.logical_not(mask), selection)
            mask = torch.logical_or(mask, selection)
            input_features[enum] = features[enum][selection]

    return input_features


class SuccessiveHalvingWrapper(sherpa.algorithms.SuccessiveHalving):
    def _get_best_vals(self, results, rung, lower_is_better):
        candidates = results[(results.rung == rung)]

        candidates_running = candidates[
            ~candidates.save_to.isin(candidates[candidates.Status == sherpa.TrialStatus.COMPLETED].save_to.values)
        ]
        candidates_old = SuccessiveHalvingWrapper._get_completed_results(results, rung)

        best_running = candidates_running.Objective.sort_values(ascending=lower_is_better).values
        best_running = None if len(best_running) <= 0 else best_running[0]
        best_completed = candidates_old.Objective.sort_values(ascending=lower_is_better).values
        best_completed = None if len(best_completed) <= 0 else best_completed[0]

        return best_running, best_completed

    def _start_new_trial(self, results, rung, promotable, lower_is_better):
        results_k = results[(results.rung == rung)].copy()
        results_k_plus = results[results.rung == rung + 1]
        results_k_plus = results_k_plus[results_k_plus.Status == sherpa.TrialStatus.COMPLETED].load_from.values

        # Failed trials should be ignore as these will never be resumed.
        failed = results_k[results_k.Status == sherpa.TrialStatus.FAILED].save_to.values
        results_k = results_k[~results_k.save_to.isin(failed)]

        # Find best value if any exists.
        best_result_k = results_k.sort_values(by="Objective", ascending=lower_is_better).save_to.values
        best_result_k = None if len(best_result_k) <= 0 else best_result_k[0]

        # If the best performing has not completed its run in the next run, start new trial, as we want to allow best
        # performing to complete before promoting at rung k+1.
        start_new = best_result_k not in results_k_plus

        # If we can promote at cur level, promote as it is fine to promote multiple from current level.
        return start_new

    def get_job(self, parameters, results, lower_is_better):
        """
        Check to see if there is a promotable configuration. Otherwise,
        return a new configuration.
        """
        all_promotable = {}
        for k in range(self.number_of_rungs):
            candidates = self.top_n(parameters,
                                    results,
                                    lower_is_better,
                                    rung=k,
                                    eta=self.eta)
            # print("RUNG", k, "CANDIDATES\n", candidates)
            promotable = candidates[~candidates.save_to.isin(
                self.promoted_trials)]
            all_promotable[k] = promotable.to_dict('records')

            # if running or completed trial has better performance than completed for rung + 1, then add new trial.
            if len(results) > 0 and self._start_new_trial(results, k, promotable, lower_is_better):
                # logger.debug(f'Currently best running as {k}')
                # start_new_trial = True
                # continue
                has_promotable = [len(all_promotable.get(i, [])) > 0 for i in range(k + 1)]

                if any(has_promotable):
                    index = has_promotable[::-1].index(True)
                    inner_k = k-index
                    promotable = all_promotable[inner_k]
                    # logger.info(f'Promoting at {inner_k}, {index}, {has_promotable}')
                    self.promoted_trials.add(promotable[0]['save_to'])
                    return promotable[0], inner_k + 1
                else:
                    # logger.info('Starting new')
                    new_config = self.rs.get_suggestion(parameters=parameters)
                    return new_config, 0

            # if len(promotable) > 0:
            #     promotable = promotable.to_dict('records')
            #     self.promoted_trials.add(promotable[0]['save_to'])
            #     return promotable[0], k+1
        # No promotable configuration found.
        else:
            # logger.info('Using default')
            return super().get_job(parameters, results, lower_is_better)
            # new_config = self.rs.get_suggestion(parameters=parameters)
            # return new_config, 0
