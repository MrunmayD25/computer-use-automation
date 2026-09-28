# Rules

Permission, privacy, record identity, correct values, intended effects, and
duplicate prevention take priority over progress. If automation cannot meet
one of these requirements, it gathers evidence within its permissions, asks
for help, or stops. Human approval cannot override a denial.

Each rule has a status. `enforced` means the required checks exist at every
relevant boundary. `partial` identifies a remaining gap. Tests use
`@pytest.mark.rule(N)` to name the rules they check. `tests/test_rules.py`
rejects rules without tests, markers for unknown rules, and enforced rules
covered only by expected failures. Expected failures must be strict.

## Rule 1. Only the profile grants permission

Missing or unrecognized declarations are rejected. Learned restrictions can
only reduce permission. Denied routes override allowed routes, and human
approval cannot grant permission that the profile withholds. Discovery
requires `environment: sandbox`.

Status: enforced.

Tests: `test_loop.py::test_discovery_runs_only_against_a_declared_sandbox`.

## Rule 2. Discovery and replay interpret evidence consistently

`matching.py` defines shared matching rules for type, purpose, cardinality,
spacing, case, record relation, and displayed versus editable values.
Screenshot text matching may ignore case and spacing, but must not merge
distinct identifiers.

Status: partial. Matching treats brackets as edge punctuation, so the
accounting amount "(4.00)" loses its negative sign and becomes "4.00".

Tests: `test_replay_regressions.py::test_contains_matches_whole_words_only`,
`test_goal_requirements.py::test_a_negated_state_does_not_contain_the_required_state`.

## Rule 3. Recording preserves the evidence proved during discovery

A saved check retains the meaning evaluated during discovery. Export may
remove or replace a check only if the remaining evidence proves the same
requirement for the same record. Otherwise, the step needs a checked human
continuation, or the export remains incomplete.

Status: enforced.

Tests: `test_recording_bridge.py::test_final_checks_preserve_live_record_evidence`,
`test_recording_bridge.py::test_a_final_check_whose_record_cannot_be_saved_is_a_gap`.

## Rule 4. Changes to the step sequence require evidence

Adding, removing, replacing, or reordering a step requires evidence that the
operation, dependencies, and policy requirements remain intact. Equal sample
values and unchanged screenshots do not prove this.

Status: partial. Recording does not detect background effects of change
handlers.

Tests: `test_recording_bridge.py::test_the_click_that_chose_the_field_is_kept`,
`test_recording_bridge.py::test_an_unchanged_next_target_does_not_remove_the_previous_click`.

## Rule 5. An invocation sends each intended business change at most once

Delivery history survives retries, graph revisits, and human interventions.
A click and an Enter keypress that submit the same change share one
operation identity. Operator-declared aliases share that identity too.
Uncertain delivery requires human help and is never resent automatically.

Status: partial. A new invocation after a stop or crash relies on the
application's duplicate key. There is no recovery contract across invocations.

Tests: `test_replay.py::test_uncertain_delivery_goes_to_a_person_and_is_never_resent`,
`test_capability.py::test_a_change_is_sent_once_across_graph_visits`.

## Rule 6. Authorization applies to the operation at dispatch

Before dispatch, the adapter rechecks the target, submission behavior, form
identity, record, and context. Any change invalidates prior authorization.
An unchanged element identifier is insufficient.

Status: partial. Declared context cannot prove hidden application handlers.
Screenshot typing checks pixels only before typing starts. Route checks use
only the address path.

Tests: `test_regressions.py::test_a_submission_changed_after_the_gate_is_not_sent`,
`test_screen.py::test_a_screen_point_names_the_observed_control_under_it`.

## Rule 7. Record-bound actions require evidence tied to their targets

Discovery, replay, and dispatch check the required record relation. A
heading or route alone is insufficient. Evidence must distinguish the
intended record from neighboring records. A row alias must identify its
column.

Navigation that changes no data proves the record before the click. Checks
on the destination page can then confirm the step without requiring the
original row to remain visible.

Status: enforced.

Tests: `test_policy.py::test_a_record_bound_action_without_evidence_is_denied`,
`test_claims.py::test_evidence_naming_another_record_is_still_refused`.

## Rule 8. Saving website text requires evidence and permission

A comparison replay confirms only text observed for a different record.
It requires consent and stops before its first data-changing step. It
neither changes data nor requests approval for a change. Repeated text is
not, by itself, proof that the text contains no customer data. Human
approval cannot authorize saving customer data.

Status: partial. An effect label comes from the run, not the page. A
comparison proves only that an operation ran again with that label.

Tests: `test_confirmation.py::test_comparison_confirms_only_text_it_saw`,
`test_confirmation.py::test_a_comparison_never_asks_a_person_to_approve_a_change`,
`test_model.py::test_value_bearing_effects_are_refused_before_proposing_input`.

## Rule 9. Values retain their source through recording and replay

Customer and invocation values remain input, fact, or secret references.
Equal sample strings do not make references interchangeable. A value with
more than one possible source is rejected.

Discovery inputs come from `--input`, the goal's text, or a person's answer
added to the goal. The model proposes values. A choice must belong to the
contract's allowed set.

Status: partial. Recording infers a value's source from equal strings
instead of receiving explicit provenance from the discovery loop.

Tests: `test_recorder.py::test_text_holding_an_input_once_is_saved_as_that_input_alone`,
`test_recording_bridge.py::test_a_text_that_is_both_an_input_and_a_kept_fact_is_refused`,
`test_inputs.py::test_a_value_the_goal_does_not_write_is_asked_of_a_person`.

## Rule 10. Screenshot values require two matching readings

A value or condition read from a screenshot counts only if two captures
agree. Disagreement leaves the result unresolved. Agreement detects screen
changes but does not prove that text recognition was correct.

Status: enforced.

Tests: `test_replay.py::test_a_painted_value_is_taken_only_when_two_captures_agree`,
`test_regressions.py::test_a_painted_line_that_changed_since_the_capture_is_not_read`.

## Rule 11. Facts retain their source and extraction rules

Whole-text and partial-text reads remain distinct during refresh and export.
Replay refreshes a changeable fact wherever discovery refreshed it. Values
over the length limit are rejected without truncation.

Status: enforced.

Tests: `test_regressions.py::test_a_partial_fact_keeps_its_value_when_read_again`,
`test_recording_bridge.py::test_a_changing_value_is_read_again_on_replay`.

## Rule 12. Execution preserves the capability and excludes values from saved logs

Approvals, interventions, and invocation data belong to the current run.
Replay never writes them into the saved capability. Command logs, journals,
redirected CLI output, and inspection diagnostics contain structural metadata,
categories, and field placeholders. Free text and screenshots stay out.
Undeclared output names receive anonymous names. Live operator displays retain
actual values. These rules apply to newly generated evidence.

Status: enforced.

Tests: `test_replay.py::test_replay_interventions_leave_the_capability_unchanged`,
`test_loop.py::test_journal_records_policy_metadata_and_never_values`,
`test_capability_cli.py::test_installed_replay_posts_once_without_a_model_and_keeps_values_private`,
`test_privacy.py::test_unknown_python_and_native_diagnostics_never_reach_saved_output`.

## Rule 13. Evaluation checks business effects, safety, and provenance

A successful write requires exactly one intended new account with the
correct values, record, operator, and institution. The run must dispatch
one commit and leave every other account unchanged. Each fault case defines
its allowed endings. A person's answer alone does not prove success.
The integrated evaluator records the revision and execution-source hashes
in newly generated run manifests. Retained evidence manifests verify file
integrity.

Status: enforced.

Tests: `test_evaluation.py::test_a_write_passes_only_with_one_intended_account_and_no_other_change`,
`test_evaluation.py::test_checked_in_capabilities_load_under_the_current_schema`.

## Rule 14. Control owns session authority

Automation sends input only while it owns the session. Human takeover
invalidates pending authorization but preserves delivery history. Commands
must identify the current run and request. Resume grants no approval.

Status: enforced.

Tests: `test_control.py::test_resume_while_an_approval_waits_declines_it_and_never_approves`,
`test_control.py::test_an_order_sent_from_an_older_status_is_stale`.

## Rule 15. Completion evidence must cover the requested task

Every required output and condition needs evidence for the intended record
and context. Passing checks that leave part of the task unproved require
human intervention, with the gaps named. Claims of absence require a complete
observation.

Status: partial. If the next target in a learned not-found branch names no
input, only the route and final checks distinguish the paths.

Tests: `test_claims.py::test_record_not_found_is_refused_while_a_result_row_shows_the_record`,
`test_claims.py::test_a_painted_answer_that_names_the_searched_record_proves_not_found`.

## Rule 16. Human interventions state their requirements and check resumption

Each request names the operation, invocation values, and question. After an
intervention, replay checks fresh evidence against saved continuation
conditions. Resume proves neither completion nor permission.

Status: partial. A generic manual step has no typed requirement covering
every possible task.

Tests: `test_replay.py::test_a_request_names_the_operation_the_record_and_the_question`,
`test_replay.py::test_a_person_who_finished_the_uncertain_submission_lets_the_replay_go_on`.

## Rule 17. Execution and recovery have explicit limits

Decisions, model requests, retries, refusals, graph traversals, and human
waits have budgets. Failed attempts count against those budgets. Harmless
actions between failures cannot reset them.

Status: enforced.

Tests: `test_loop.py::test_step_budget_stops_the_run`,
`test_regressions.py::test_repeated_findings_hand_the_run_to_a_person`.

## Rule 18. Secrets and manual input are protected before observation

Structured observations, screenshots, reads, and diagnostics apply the same
protection. If a channel cannot establish that protection, it refuses the
observation.

Status: enforced for the browser, the only supported adapter.

Tests: `test_browser.py::test_a_screenshot_masks_password_fields`,
`test_regressions.py::test_a_secret_field_is_never_read_by_read_or_assert`.

## Rule 19. Observation and read permissions are distinct

Observing content, extracting a value, and evaluating a condition require
separate permissions. Discovery and replay apply those permissions
consistently, including when checks compare text.

Status: enforced.

Tests: `test_discovery.py::test_an_observation_mode_the_profile_denies_is_refused`,
`test_read_contract.py::test_a_branch_check_that_compares_text_needs_the_read_grant`.

## Rule 20. Evaluation data stays out of product code

Values from evaluation sites and goals must not appear in `src/computeruse`,
including prompts, comments, and docstrings. Profiles and contracts supply
site context. Evaluation runs must not change site behavior.

Status: enforced for values listed in
`evaluation/sites/fixture-values.yaml`.

Tests: `test_rules.py::test_no_evaluation_value_appears_in_product_source`.
