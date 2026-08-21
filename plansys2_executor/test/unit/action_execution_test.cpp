// Copyright 2021 Intelligent Robotics Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "plansys2_domain_expert/DomainExpertNode.hpp"
#include "plansys2_domain_expert/DomainExpertClient.hpp"
#include "plansys2_problem_expert/ProblemExpertNode.hpp"
#include "plansys2_problem_expert/ProblemExpertClient.hpp"
#include "plansys2_planner/PlannerNode.hpp"
#include "plansys2_planner/PlannerClient.hpp"
#include "plansys2_executor/BTBuilder.hpp"

#include "plansys2_executor/ActionExecutor.hpp"
#include "plansys2_executor/ActionExecutorClient.hpp"
#include "plansys2_executor/ExecutorNode.hpp"
#include "plansys2_executor/ExecutorClient.hpp"
#include "plansys2_problem_expert/Utils.hpp"

#include "behaviortree_cpp/behavior_tree.h"
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/utils/shared_library.h"
#include "behaviortree_cpp/blackboard.h"

#include "plansys2_executor/behavior_tree/execute_action_node.hpp"
#include "plansys2_executor/behavior_tree/wait_action_node.hpp"
#include "plansys2_executor/behavior_tree/wait_atstart_req_node.hpp"
#include "plansys2_executor/behavior_tree/check_overall_req_node.hpp"
#include "plansys2_executor/behavior_tree/check_atend_req_node.hpp"
#include "plansys2_executor/behavior_tree/apply_atstart_effect_node.hpp"
#include "plansys2_executor/behavior_tree/apply_atend_effect_node.hpp"

#include "lifecycle_msgs/msg/state.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "gtest/gtest.h"


using CallbackReturnT =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using namespace std::chrono_literals;

class MoveAction : public plansys2::ActionExecutorClient
{
public:
  using Ptr = std::shared_ptr<MoveAction>;
  static Ptr make_shared(const std::string & node_name, const std::chrono::nanoseconds & rate)
  {
    return std::make_shared<MoveAction>(node_name, rate);
  }


  MoveAction(const std::string & id, const std::chrono::nanoseconds & rate)
  : ActionExecutorClient(id, rate)
  {
    executions_ = 0;
    cycles_ = 0;
  }

  CallbackReturnT
  on_activate(const rclcpp_lifecycle::State & state)
  {
    std::cerr << "MoveAction::on_activate" << std::endl;
    counter_ = 0;

    return ActionExecutorClient::on_activate(state);
  }

  void do_work() override
  {
    RCLCPP_INFO_STREAM(get_logger(), "Executing [" << action_managed_ << "]");
    for (const auto & arg : current_arguments_) {
      RCLCPP_INFO_STREAM(get_logger(), "\t[" << arg << "]");
    }

    cycles_++;

    if (counter_++ > 3) {
      finish(true, 1.0, "completed");
      executions_++;
    } else {
      send_feedback(counter_ * 0.0, "running");
    }
  }

  int counter_;
  int executions_;
  int cycles_;
};

class ActionExecutorTestAdapter : public plansys2::ActionExecutor
{
public:
  using Ptr = std::shared_ptr<ActionExecutorTestAdapter>;

  static Ptr make_shared(
    const std::string & action,
    rclcpp_lifecycle::LifecycleNode::SharedPtr node)
  {
    return std::make_shared<ActionExecutorTestAdapter>(action, node);
  }

  ActionExecutorTestAdapter(
    const std::string & action,
    rclcpp_lifecycle::LifecycleNode::SharedPtr node)
  : ActionExecutor(action, node)
  {
  }

  void expire_retry_period()
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    last_request_time_ = std::chrono::steady_clock::now() - 2s;
  }

  void expire_dealing_timeout()
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_time_ = node_->now() - rclcpp::Duration::from_seconds(31.0);
  }
};

class ActionExecutorRetryTest : public testing::Test
{
protected:
  using ActionExecution = plansys2_msgs::msg::ActionExecution;

  void SetUp() override
  {
    test_node_ = rclcpp::Node::make_shared("action_executor_retry_test_node");
    test_lf_node_ =
      rclcpp_lifecycle::LifecycleNode::make_shared("action_executor_retry_test_lf_node");

    action_hub_sub_ = test_node_->create_subscription<ActionExecution>(
      "/actions_hub", rclcpp::QoS(100).reliable(),
      [this](const ActionExecution::SharedPtr msg) {messages_.push_back(*msg);});
    action_hub_pub_ = test_node_->create_publisher<ActionExecution>(
      "/actions_hub", rclcpp::QoS(100).reliable());

    executor_.add_node(test_node_);
    executor_.add_node(test_lf_node_->get_node_base_interface());
  }

  ActionExecutorTestAdapter::Ptr make_action_executor()
  {
    auto action_executor = ActionExecutorTestAdapter::make_shared(
      "(move r2d2 steering_wheels_zone assembly_zone)", test_lf_node_);
    spin_some_for(100ms);
    return action_executor;
  }

  void spin_some_for(std::chrono::milliseconds duration = 50ms)
  {
    const auto end = std::chrono::steady_clock::now() + duration;
    do {
      executor_.spin_some();
      std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < end);
  }

  void publish(
    const ActionExecutorTestAdapter & action_executor,
    int16_t type,
    bool success = false)
  {
    ActionExecution msg;
    msg.type = type;
    msg.node_id = "performer";
    msg.action = action_executor.get_action_name();
    msg.arguments = action_executor.get_action_params();
    msg.success = success;
    msg.completion = success ? 1.0 : 0.0;
    action_hub_pub_->publish(msg);
    spin_some_for();
  }

  size_t count_messages(int16_t type) const
  {
    return std::count_if(
      messages_.begin(), messages_.end(),
      [type](const ActionExecution & msg) {return msg.type == type;});
  }

  rclcpp::Node::SharedPtr test_node_;
  rclcpp_lifecycle::LifecycleNode::SharedPtr test_lf_node_;
  rclcpp::Subscription<ActionExecution>::SharedPtr action_hub_sub_;
  rclcpp::Publisher<ActionExecution>::SharedPtr action_hub_pub_;
  std::vector<ActionExecution> messages_;
  rclcpp::experimental::executors::EventsExecutor executor_;
};

TEST_F(ActionExecutorRetryTest, retry_is_driven_only_by_dealing_state)
{
  auto action_executor = make_action_executor();

  ASSERT_EQ(action_executor->tick(test_node_->now()), BT::NodeStatus::RUNNING);
  spin_some_for();
  ASSERT_EQ(count_messages(ActionExecution::REQUEST), 1u);

  action_executor->tick(test_node_->now());
  spin_some_for();
  ASSERT_EQ(count_messages(ActionExecution::REQUEST), 1u);

  action_executor->expire_retry_period();
  action_executor->tick(test_node_->now());
  spin_some_for();
  ASSERT_EQ(count_messages(ActionExecution::REQUEST), 2u);

  action_executor->expire_dealing_timeout();
  ASSERT_EQ(action_executor->tick(test_node_->now()), BT::NodeStatus::FAILURE);
  action_executor->expire_retry_period();
  action_executor->tick(test_node_->now());
  spin_some_for();

  EXPECT_EQ(
    action_executor->get_internal_status(), plansys2::ActionExecutor::Status::FAILURE);
  EXPECT_EQ(count_messages(ActionExecution::REQUEST), 2u);
}

TEST_F(ActionExecutorRetryTest, terminal_state_disarms_retry_and_next_action_dispatches)
{
  auto first_executor = make_action_executor();
  auto second_executor = make_action_executor();

  first_executor->tick(test_node_->now());
  spin_some_for();
  publish(*first_executor, ActionExecution::RESPONSE);
  publish(*first_executor, ActionExecution::FINISH, true);

  ASSERT_EQ(
    first_executor->get_internal_status(), plansys2::ActionExecutor::Status::SUCCESS);
  first_executor->expire_retry_period();
  first_executor->tick(test_node_->now());
  spin_some_for();
  ASSERT_EQ(count_messages(ActionExecution::REQUEST), 1u);

  second_executor->tick(test_node_->now());
  spin_some_for();
  publish(*second_executor, ActionExecution::RESPONSE);
  publish(*second_executor, ActionExecution::FINISH, true);

  EXPECT_EQ(
    second_executor->get_internal_status(), plansys2::ActionExecutor::Status::SUCCESS);
  EXPECT_EQ(count_messages(ActionExecution::REQUEST), 2u);
  EXPECT_EQ(count_messages(ActionExecution::CONFIRM), 2u);
  EXPECT_EQ(count_messages(ActionExecution::REJECT), 0u);
}

TEST_F(ActionExecutorRetryTest, cancellation_is_terminal_for_late_messages)
{
  auto action_executor = make_action_executor();

  action_executor->tick(test_node_->now());
  spin_some_for();
  publish(*action_executor, ActionExecution::RESPONSE);
  ASSERT_EQ(
    action_executor->get_internal_status(), plansys2::ActionExecutor::Status::RUNNING);

  action_executor->cancel();
  spin_some_for();
  publish(*action_executor, ActionExecution::FINISH, true);

  action_executor->expire_retry_period();
  action_executor->tick(test_node_->now());
  spin_some_for();

  EXPECT_EQ(
    action_executor->get_internal_status(), plansys2::ActionExecutor::Status::CANCELLED);
  EXPECT_EQ(count_messages(ActionExecution::REQUEST), 1u);
  EXPECT_EQ(count_messages(ActionExecution::CANCEL), 1u);
}

TEST(action_execution, protocol_basic)
{
  auto test_node = rclcpp::Node::make_shared("test_node");
  auto test_lf_node = rclcpp_lifecycle::LifecycleNode::make_shared("test_lf_node");
  auto move_action_node = std::make_shared<MoveAction>("move_action", 1s);
  auto move_action_executor = plansys2::ActionExecutor::make_shared(
    "(move r2d2 steering_wheels_zone assembly_zone)", test_lf_node);

  ASSERT_EQ(move_action_executor->get_action_name(), "move");
  ASSERT_EQ(move_action_executor->get_action_params().size(), 3u);
  ASSERT_EQ(move_action_executor->get_action_params()[0], "r2d2");
  ASSERT_EQ(move_action_executor->get_action_params()[2], "assembly_zone");

  move_action_node->set_parameter({"action_name", "move"});

  rclcpp::experimental::executors::EventsExecutor exe;

  exe.add_node(test_node);
  exe.add_node(test_lf_node->get_node_base_interface());
  exe.add_node(move_action_node->get_node_base_interface());

  std::vector<plansys2_msgs::msg::ActionExecution> action_execution_msgs;

  auto action_hub_sub = test_node->create_subscription<plansys2_msgs::msg::ActionExecution>(
    "/actions_hub", rclcpp::QoS(100).reliable(),
    [&action_execution_msgs](const plansys2_msgs::msg::ActionExecution::SharedPtr msg) {
      action_execution_msgs.push_back(*msg);
    });

  bool finish = false;
  std::thread t([&]() {
      while (!finish) {exe.spin_some();}
    });

  ASSERT_EQ(move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::IDLE);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::NOT_READY);

  test_lf_node->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  move_action_node->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 0.5) {
      rate.sleep();
    }
  }

  ASSERT_EQ(
    move_action_node->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::READY);
  ASSERT_TRUE(action_execution_msgs.empty());

  {
    std::vector<BT::NodeStatus> tick_status_log;
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 0.5) {
      tick_status_log.push_back(move_action_executor->tick(test_node->now()));
      rate.sleep();
    }
    ASSERT_EQ(tick_status_log[0], BT::NodeStatus::RUNNING);
  }

  ASSERT_EQ(
    move_action_node->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  ASSERT_EQ(move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::RUNNING);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::RUNNING);

  ASSERT_EQ(action_execution_msgs.size(), 3u);
  ASSERT_EQ(action_execution_msgs[0].type, plansys2_msgs::msg::ActionExecution::REQUEST);
  ASSERT_EQ(action_execution_msgs[1].type, plansys2_msgs::msg::ActionExecution::RESPONSE);
  ASSERT_EQ(action_execution_msgs[2].type, plansys2_msgs::msg::ActionExecution::CONFIRM);

  {
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 5) {
      move_action_executor->tick(test_node->now());
      rate.sleep();
    }
  }

  ASSERT_EQ(move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::SUCCESS);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::READY);


  ASSERT_EQ(action_execution_msgs.size(), 8u);
  ASSERT_EQ(action_execution_msgs[3].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[4].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[5].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[6].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[7].type, plansys2_msgs::msg::ActionExecution::FINISH);


  ASSERT_EQ(move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::SUCCESS);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::READY);

  finish = true;
  t.join();
}

TEST(action_execution, protocol_cancelation)
{
  auto test_node = rclcpp::Node::make_shared("test_node");
  auto test_lf_node = rclcpp_lifecycle::LifecycleNode::make_shared("test_lf_node");
  auto move_action_node = std::make_shared<MoveAction>("move_action", 1s);
  auto move_action_executor = plansys2::ActionExecutor::make_shared(
    "(move r2d2 steering_wheels_zone assembly_zone)", test_lf_node);

  ASSERT_EQ(move_action_executor->get_action_name(), "move");
  ASSERT_EQ(move_action_executor->get_action_params().size(), 3u);
  ASSERT_EQ(move_action_executor->get_action_params()[0], "r2d2");
  ASSERT_EQ(move_action_executor->get_action_params()[2], "assembly_zone");

  move_action_node->set_parameter({"action_name", "move"});

  rclcpp::experimental::executors::EventsExecutor exe;

  exe.add_node(test_node);
  exe.add_node(test_lf_node->get_node_base_interface());
  exe.add_node(move_action_node->get_node_base_interface());

  std::vector<plansys2_msgs::msg::ActionExecution> action_execution_msgs;

  auto action_hub_sub = test_node->create_subscription<plansys2_msgs::msg::ActionExecution>(
    "/actions_hub", rclcpp::QoS(100).reliable(),
    [&action_execution_msgs](const plansys2_msgs::msg::ActionExecution::SharedPtr msg) {
      action_execution_msgs.push_back(*msg);
    });

  bool finish = false;
  std::thread t([&]() {
      while (!finish) {exe.spin_some();}
    });

  ASSERT_EQ(move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::IDLE);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::NOT_READY);

  test_lf_node->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  move_action_node->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 0.5) {
      rate.sleep();
    }
  }

  ASSERT_EQ(
    move_action_node->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::READY);
  ASSERT_TRUE(action_execution_msgs.empty());

  {
    std::vector<BT::NodeStatus> tick_status_log;
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 0.5) {
      tick_status_log.push_back(move_action_executor->tick(test_node->now()));
      rate.sleep();
    }
    ASSERT_EQ(tick_status_log[0], BT::NodeStatus::RUNNING);
  }

  ASSERT_EQ(
    move_action_node->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  ASSERT_EQ(
    move_action_executor->get_internal_status(), plansys2::ActionExecutor::Status::RUNNING);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::RUNNING);

  ASSERT_EQ(action_execution_msgs.size(), 3u);
  ASSERT_EQ(action_execution_msgs[0].type, plansys2_msgs::msg::ActionExecution::REQUEST);
  ASSERT_EQ(action_execution_msgs[1].type, plansys2_msgs::msg::ActionExecution::RESPONSE);
  ASSERT_EQ(action_execution_msgs[2].type, plansys2_msgs::msg::ActionExecution::CONFIRM);

  {
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 2) {
      move_action_executor->tick(test_node->now());
      rate.sleep();
    }
  }

  move_action_executor->cancel();

  {
    rclcpp::Rate rate(10);
    auto start = test_node->now();
    while ((test_node->now() - start).seconds() < 2) {
      move_action_executor->tick(test_node->now());
      rate.sleep();
    }
  }

  ASSERT_EQ(
    move_action_executor->get_internal_status(),
    plansys2::ActionExecutor::Status::CANCELLED);
  ASSERT_EQ(
    move_action_node->get_internal_status().state,
    plansys2_msgs::msg::ActionPerformerStatus::READY);


  ASSERT_EQ(action_execution_msgs.size(), 6u);
  ASSERT_EQ(action_execution_msgs[3].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[4].type, plansys2_msgs::msg::ActionExecution::FEEDBACK);
  ASSERT_EQ(action_execution_msgs[5].type, plansys2_msgs::msg::ActionExecution::CANCEL);

  finish = true;
  t.join();
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);

  return RUN_ALL_TESTS();
}
