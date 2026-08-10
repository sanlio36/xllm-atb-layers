/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/operation_infra.h>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

enum class CaptureEventAction : int32_t {
    RECORD = 0,
    WAIT = 1,
};

class CaptureEventState final {
public:
    explicit CaptureEventState(aclrtEvent event) : event_(event) {}

    CaptureEventState(const CaptureEventState &) = delete;
    CaptureEventState &operator=(const CaptureEventState &) = delete;

    ~CaptureEventState()
    {
        if (event_ == nullptr) {
            return;
        }
        const aclError ret = aclrtDestroyEvent(event_);
        if (ret != ACL_SUCCESS) {
            ATB_SPEED_LOG_ERROR("aclrtDestroyEvent failed, aclError = " << ret);
        }
    }

    aclrtEvent event() const
    {
        return event_;
    }

private:
    aclrtEvent event_ = nullptr;
};

class CaptureEventOperation final : public atb::OperationInfra {
public:
    CaptureEventOperation(std::string name, std::shared_ptr<CaptureEventState> state,
                          CaptureEventAction action, uint64_t stream_id)
        : name_(std::move(name)), state_(std::move(state)), action_(action),
          stream_id_(stream_id)
    {
    }

    std::string GetName() const override
    {
        return name_;
    }

    uint32_t GetInputNum() const override
    {
        return 1;
    }

    uint32_t GetOutputNum() const override
    {
        return 0;
    }

    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &,
                           atb::SVector<atb::TensorDesc> &) const override
    {
        return atb::NO_ERROR;
    }

    atb::Status Setup(const atb::VariantPack &, uint64_t &workspace_size,
                      atb::Context *context) override
    {
        workspace_size = 0;
        return validate_context(context);
    }

    atb::Status Execute(const atb::VariantPack &, uint8_t *, uint64_t,
                        atb::Context *context) override
    {
        CHECK_OPERATION_STATUS_RETURN(validate_context(context));
        const std::vector<aclrtStream> streams = context->GetExecuteStreams();
        aclError ret = ACL_SUCCESS;
        if (action_ == CaptureEventAction::RECORD) {
            ret = aclrtRecordEvent(state_->event(), streams.at(stream_id_));
        } else {
            ret = aclrtStreamWaitEvent(streams.at(stream_id_), state_->event());
        }
        if (ret != ACL_SUCCESS) {
            ATB_SPEED_LOG_ERROR(name_ << " failed, aclError = " << ret);
            return atb::ERROR_INVALID_PARAM;
        }
        return atb::NO_ERROR;
    }

private:
    atb::Status validate_context(atb::Context *context) const
    {
        if (context == nullptr || state_ == nullptr || state_->event() == nullptr) {
            ATB_SPEED_LOG_ERROR(name_ << " has invalid context or event");
            return atb::ERROR_INVALID_PARAM;
        }
        const std::vector<aclrtStream> streams = context->GetExecuteStreams();
        if (stream_id_ >= streams.size() || streams.at(stream_id_) == nullptr) {
            ATB_SPEED_LOG_ERROR(name_ << " cannot find execute stream " << stream_id_);
            return atb::ERROR_INVALID_PARAM;
        }
        return atb::NO_ERROR;
    }

    std::string name_;
    std::shared_ptr<CaptureEventState> state_;
    CaptureEventAction action_;
    uint64_t stream_id_ = 0;
};

inline atb::Status CreateCaptureEvent(std::shared_ptr<CaptureEventState> &state)
{
    aclrtEvent event = nullptr;
    const aclError ret = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("aclrtCreateEventExWithFlag failed, aclError = " << ret);
        return atb::ERROR_INVALID_PARAM;
    }
    state = std::make_shared<CaptureEventState>(event);
    return atb::NO_ERROR;
}

inline atb::Status AddCaptureEventNode(
    atb::GraphParam &op_graph, const std::string &name,
    const std::shared_ptr<CaptureEventState> &event,
    CaptureEventAction action, uint64_t stream_id, uint32_t input_tensor_id)
{
    if (event == nullptr || input_tensor_id == UINT32_MAX) {
        ATB_SPEED_LOG_ERROR(name << " has invalid event or input tensor id");
        return atb::ERROR_INVALID_PARAM;
    }
    atb::Node event_node;
    event_node.operation = new CaptureEventOperation(name, event, action, stream_id);
    // The input anchors host submission after the tensor producer.
    event_node.inTensorIds = {input_tensor_id};
    if (stream_id != 0) {
        CHECK_OPERATION_STATUS_RETURN(atb::SetExecuteStreamId(event_node.operation, stream_id));
    }
    op_graph.nodes.push_back(event_node);
    return atb::NO_ERROR;
}

}  // namespace common
}  // namespace atb_speed
