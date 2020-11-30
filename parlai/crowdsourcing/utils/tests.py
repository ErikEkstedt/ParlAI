#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for running tests.
"""

import os
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Sequence

from hydra.experimental import compose, initialize
from mephisto.abstractions.blueprint import SharedTaskState
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import augment_config_from_db


class AbstractCrowdsourcingTest(unittest.TestCase):
    """
    Abstract class for end-to-end tests of Mephisto-based crowdsourcing tasks.

    Allows for setup and teardown of the operator, as well as for config specification
    and agent registration.
    """

    def setUp(self):
        self.operator = None

    def tearDown(self):
        if self.operator is not None:
            self.operator.shutdown()

    def _set_up_config(
        self,
        blueprint_type: str,
        task_directory: str,
        overrides: Optional[List[str]] = None,
    ):
        """
        Set up the config and database.

        Uses the Hydra compose() API for unit testing and a temporary directory to store
        the test database.
        :param blueprint_type: string uniquely specifying Blueprint class
        :param task_directory: directory containing the `conf/` configuration folder.
          Will be injected as `${task_dir}` in YAML files.
        :param overrides: additional config overrides
        """

        # Define the configuration settings
        relative_task_directory = os.path.relpath(
            task_directory, os.path.dirname(__file__)
        )
        relative_config_path = os.path.join(relative_task_directory, 'conf')
        if overrides is None:
            overrides = []
        with initialize(config_path=relative_config_path):
            self.config = compose(
                config_name="example",
                overrides=[
                    f'+mephisto.blueprint._blueprint_type={blueprint_type}',
                    f'+mephisto/architect=mock',
                    f'+mephisto/provider=mock',
                    f'+task_dir={task_directory}',
                    f'+current_time={int(time.time())}',
                ]
                + overrides,
            )
            # TODO: when Hydra 1.1 is released with support for recursive defaults,
            #  don't manually specify all missing blueprint args anymore, but
            #  instead define the blueprint in the defaults list directly.
            #  Currently, the blueprint can't be set in the defaults list without
            #  overriding params in the YAML file, as documented at
            #  https://github.com/facebookresearch/hydra/issues/326 and as fixed in
            #  https://github.com/facebookresearch/hydra/pull/1044.

        self.data_dir = tempfile.mkdtemp()
        database_path = os.path.join(self.data_dir, "mephisto.db")
        self.db = LocalMephistoDB(database_path)
        self.config = augment_config_from_db(self.config, self.db)
        self.config.mephisto.architect.should_run_server = True

    def _set_up_server(self, shared_state: Optional[SharedTaskState] = None):
        """
        Set up the operator and server.
        """
        self.operator = Operator(self.db)
        self.operator.validate_and_run_config(
            self.config.mephisto, shared_state=shared_state
        )
        channel_info = list(self.operator.supervisor.channels.values())[0]
        self.server = channel_info.job.architect.server

    def _register_mock_agents(self, num_agents: int = 1) -> List[str]:
        """
        Register mock agents for testing, taking the place of crowdsourcing workers.

        Specify the number of agents to register. Return the agents' IDs after creation.
        """

        for idx in range(num_agents):

            # Register the worker
            mock_worker_name = f"MOCK_WORKER_{idx:d}"
            self.server.register_mock_worker(mock_worker_name)
            workers = self.db.find_workers(worker_name=mock_worker_name)
            worker_id = workers[0].db_id

            # Register the agent
            mock_agent_details = f"FAKE_ASSIGNMENT_{idx:d}"
            self.server.register_mock_agent(worker_id, mock_agent_details)

        # Get all agents' IDs
        agents = self.db.find_agents()
        agent_ids = [agent.db_id for agent in agents]

        return agent_ids


class AbstractOneTurnCrowdsourcingTest(AbstractCrowdsourcingTest):
    """
    Abstract class for end-to-end tests of one-turn crowdsourcing tasks.

    Useful for Blueprints such as AcuteEvalBlueprint and StaticReactBlueprint for which
    all of the worker's responses are sent to the backend code at once.
    """

    def _test_agent_state(self, expected_state: Dict[str, Any]):
        """
        Test that the actual agent state matches the expected state.

        Register a mock human agent, request initial data to define the 'inputs' field
        of the agent state, make the agent act to define the 'outputs' field of the
        agent state, and then check that the agent state matches the desired agent
        state.
        """

        # Set up the mock human agent
        agent_id = self._register_mock_agents(num_agents=1)[0]

        # Set initial data
        self.server.request_init_data(agent_id)

        # Make agent act
        self.server.send_agent_act(
            agent_id,
            {"MEPHISTO_is_submit": True, "task_data": expected_state['outputs']},
        )

        # Check that the inputs and outputs are as expected
        state = self.db.find_agents()[0].state.get_data()
        self.assertEqual(expected_state['inputs'], state['inputs'])
        self.assertEqual(expected_state['outputs'], state['outputs'])


class AbstractParlAIChatTest(AbstractCrowdsourcingTest):
    """
    Abstract class for end-to-end tests of one-turn ParlAIChatBlueprint tasks.
    """

    def _test_agent_states(
        self,
        num_agents: int,
        agent_display_ids: Sequence[str],
        agent_messages: List[Sequence[str]],
        form_messages: Sequence[str],
        form_task_data: Sequence[Dict[str, Any]],
        expected_states: Sequence[Dict[str, Any]],
        agent_task_data: Optional[List[Sequence[Dict[str, Any]]]] = None,
    ):
        """
        Test that the actual agent states match the expected states.

        Register mock human agents, request initial data to define the 'inputs' fields
        of the agent states, make the agents have a conversation to define the 'outputs'
        fields of the agent states, and then check that the agent states all match the
        desired agent states.
        """

        # If no task data was supplied, create empty task data
        if agent_task_data is None:
            agent_task_data = []
            for message_round in agent_messages:
                agent_task_data.append([{}] * len(message_round))

        # Set up the mock human agents
        agent_ids = self._register_mock_agents(num_agents=num_agents)

        # # Feed messages to the agents

        # Set initial data
        for agent_id in agent_ids:
            self.server.request_init_data(agent_id)

        # Have agents talk to each other
        assert len(agent_messages) == len(agent_task_data)
        for message_round, task_data_round in zip(agent_messages, agent_task_data):
            assert len(message_round) == len(task_data_round) == len(agent_ids)
            for agent_id, agent_display_id, message, task_data in zip(
                agent_ids, agent_display_ids, message_round, task_data_round
            ):
                self._send_agent_message(
                    agent_id=agent_id,
                    agent_display_id=agent_display_id,
                    text=message,
                    task_data=task_data,
                )

        # Have agents fill out the form
        for agent_idx, agent_id in enumerate(agent_ids):
            self.server.send_agent_act(
                agent_id=agent_id,
                act_content={
                    'text': form_messages[agent_idx],
                    'task_data': form_task_data[agent_idx],
                    'id': agent_display_ids[agent_idx],
                    'episode_done': False,
                },
            )

        # Submit the HIT
        for agent_id in agent_ids:
            self.server.send_agent_act(
                agent_id=agent_id,
                act_content={
                    'task_data': {'final_data': {}},
                    'MEPHISTO_is_submit': True,
                },
            )

        # # Check that the inputs and outputs are as expected

        actual_states = [agent.state.get_data() for agent in self.db.find_agents()]
        assert len(actual_states) == len(expected_states)
        for actual_state, desired_state in zip(actual_states, expected_states):
            assert actual_state['inputs'] == desired_state['inputs']
            assert len(actual_state['outputs']['messages']) == len(
                desired_state['outputs']['messages']
            )
            for actual_message, desired_message in zip(
                actual_state['outputs']['messages'],
                desired_state['outputs']['messages'],
            ):
                for key, desired_value in desired_message.items():
                    if key == 'timestamp':
                        pass  # The timestamp will obviously be different
                    elif key == 'data':
                        for key_inner, desired_value_inner in desired_message[
                            key
                        ].items():
                            if key_inner in ['beam_texts', 'message_id']:
                                pass  # The message ID will be different
                            else:
                                self.assertEqual(
                                    actual_message[key][key_inner], desired_value_inner
                                )
                    else:
                        self.assertEqual(actual_message[key], desired_value)

    def _send_agent_message(
        self, agent_id: str, agent_display_id: str, text: str, task_data: Dict[str, Any]
    ):
        """
        Have the agent specified by agent_id send the specified text and task data with
        the given display ID string.
        """
        act_content = {
            "text": text,
            "task_data": task_data,
            "id": agent_display_id,
            "episode_done": False,
        }
        self.server.send_agent_act(agent_id=agent_id, act_content=act_content)
