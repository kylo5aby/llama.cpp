@llama.cpp
@server
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   BOS token is 1
    And   42 as server seed
      # KV Cache corresponds to the total amount of tokens
      # that can be stored across all independent sequences: #4130
      # see --ctx-size and #5568
    And   256 KV cache size
    And   32 as batch size
    And   2 slots
    # And   64 server max tokens to predict
    And   prometheus compatible metrics exposed
    Then  the server is starting
    Then  the server is healthy

  Scenario: Health
    Then the server is ready
    And  all slots are idle


  Scenario Outline: Completion
    Given a prompt <prompt>
    And   <n_predict> max tokens to predict
    And   a completion request with no api error
    Then  <n_predicted> tokens are predicted matching <re_content>
    And   the completion is <truncated> truncated
    And   <n_prompt> prompt tokens are processed
    And   prometheus metrics are exposed
    And   metric llamacpp:tokens_predicted is <n_predicted>

    Examples: Prompts
      | prompt                                                                    | n_predict | re_content                                  | n_prompt | n_predicted | truncated |
      | Write a joke about AI from a very long prompt which will not be truncated | -2        | (princesses\|everyone\|kids\|Anna\|forest)+ | 46       | 256         | not       |

