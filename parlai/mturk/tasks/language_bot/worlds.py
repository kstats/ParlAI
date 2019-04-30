#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed

import os
import time
import pickle
import numpy as np

class MTurkLangDialogOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        self.mturk_agent.observe({
            'id': 'System',
            'text': 'Welcome onboard!'
        })
        self.mturk_agent.act()
        self.mturk_agent.observe({
            'id': 'System',
            'text': 'Thank you for your input! Please wait while '
                    'we match you with another worker...'
        })
        self.episodeDone = True


class MTurkLangDialogWorld(MTurkTaskWorld):
    """Basic world where each agent gets a turn in a round-robin fashion,
    receiving as input the actions of all other agents since that agent last
    acted.
    """
    def __init__(self, opt, agents=None, shared=None):
        #Let's save the task type for the purpose of saving
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        #Let's keep track of dialogue for the purpose of saving
        self.dialog = []
        #path to save data
        self.data_path = "./data"
        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False


    def parley(self):
        """For each agent, get an observation of the last action each of the
        other agents took. Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=None)
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent
            if acts[index]['episode_done']:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
            self.dialog.append((index, acts[index]['text']))

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from
        completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.agents)
    def save_data(self):
        # save persona_idx_stack
        convo_finished = True
        bad_workers = []
        for ag in self.agents:
            if (ag.hit_is_abandoned or ag.hit_is_returned or
                    ag.disconnected or ag.hit_is_expired):
                bad_workers.append(ag.worker_id)
                convo_finished = False
        if not convo_finished or self.dialog == []:
            for ag in self.agents:
                ag.not_approve = True


        data_path = self.data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000), self.task_type
                )
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        print(self.world_tag + ': Data successfully saved at {}.'.format(filename))
        pickle.dump({'dialog': self.dialog,
                     'workers': [ag.worker_id for ag in self.agents],
                     'bad_workers': bad_workers}, open(filename, 'wb'))