"""Play page - Game mode and challenge modes.

Contains two tabs:
  - **Decode My Brain**: guess movement direction from spike patterns
    and compete against a model decoder.
  - **Challenge Modes**: timed/scored competitive modes with leaderboards
    and achievements.
"""

import numpy as np
import streamlit as st

from simulation import simulate_trial
from decoders import PopulationVectorDecoder, MaximumLikelihoodDecoder
from engine.game import GameEngine
from challenges import (
    ChallengeMode,
    CHALLENGE_CONFIGS,
    SCORING_DESCRIPTIONS,
    score_breakdown,
)
from visualization import (
    plot_population_bar,
    plot_polar_comparison,
    create_scoreboard_table,
    plot_leaderboard,
)
from utils import radians_to_degrees, angular_error_degrees


st.session_state.visited_play = True

if not st.session_state.get('simulated', False):
    st.warning("Please run a simulation using the sidebar controls.")
    st.stop()

neurons = st.session_state.neurons

tab_game, tab_challenge = st.tabs(["Decode My Brain", "Challenge Modes"])


# =========================================================================
# Game Mode
# =========================================================================

with tab_game:
    st.header("Decode My Brain")
    st.markdown("""
    **Can you decode movement direction from neural activity?**

    Look at the spike pattern and try to guess the hidden movement direction.
    Compare your accuracy to the model decoder!
    """)

    pv_decoder = PopulationVectorDecoder()
    ml_decoder = MaximumLikelihoodDecoder()

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Game Controls")

        decoder_choice = st.radio(
            "Model decoder",
            ["Population Vector", "Maximum Likelihood"],
            help="Which decoder algorithm the model uses"
        )
        selected_decoder = pv_decoder if decoder_choice == "Population Vector" else ml_decoder

        if st.button("New Round", type="primary", use_container_width=True):
            theta, spikes = GameEngine.generate_round(
                neurons,
                duration_ms=st.session_state.duration_ms,
                variance_scale=st.session_state.variance_scale,
            )
            st.session_state.game_theta = theta
            st.session_state.game_spikes = spikes
            st.session_state.game_active = True
            st.session_state.round_submitted = False
            st.session_state.last_result = None
            st.rerun()

        if st.button("Reset Game", use_container_width=True):
            st.session_state.game_rounds = []
            st.session_state.game_active = False
            st.session_state.game_theta = None
            st.session_state.game_spikes = None
            st.session_state.round_submitted = False
            st.session_state.last_result = None
            st.rerun()

        # Scoreboard
        if st.session_state.game_rounds:
            st.markdown("---")
            st.subheader("Scoreboard")

            user_errors = [r['user_error'] for r in st.session_state.game_rounds]
            model_errors = [r['model_error'] for r in st.session_state.game_rounds]
            user_wins = sum(1 for r in st.session_state.game_rounds if r['winner'] == 'User')
            model_wins = sum(1 for r in st.session_state.game_rounds if r['winner'] == 'Model')

            st.metric("Rounds Played", len(st.session_state.game_rounds))

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Your Mean Error", f"{np.mean(user_errors):.1f}°")
                st.metric("Your Wins", user_wins)
            with c2:
                st.metric("Model Mean Error", f"{np.mean(model_errors):.1f}°")
                st.metric("Model Wins", model_wins)

    with col1:
        if st.session_state.game_active and st.session_state.game_spikes is not None:
            st.subheader("What direction is this?")
            st.markdown("*Study the spike pattern below and make your guess!*")

            fig_game = plot_population_bar(
                st.session_state.game_spikes,
                neurons,
                true_theta=None,
                as_polar=True
            )
            fig_game.update_layout(title="Population Activity (Hidden Direction)")
            st.plotly_chart(fig_game, use_container_width=True)

            if not st.session_state.round_submitted:
                user_guess_deg = st.number_input(
                    "Your guess (degrees)",
                    min_value=0, max_value=359, value=180, step=1,
                    help="Enter the direction you think this activity represents (0-359)",
                    key="game_guess_input"
                )

                if st.button("Submit Guess", type="primary"):
                    game_result = GameEngine.submit_guess(
                        user_deg=user_guess_deg,
                        true_theta=st.session_state.game_theta,
                        spike_counts=st.session_state.game_spikes,
                        neurons=neurons,
                        decoder=selected_decoder,
                        duration_s=st.session_state.duration_ms / 1000,
                    )

                    result = game_result.to_dict()
                    st.session_state.game_rounds.append(result)
                    st.session_state.last_result = result
                    st.session_state.round_submitted = True
                    st.rerun()

            else:
                result = st.session_state.last_result

                st.markdown("### Results")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("True Direction", f"{result['true_deg']:.0f}°")
                with c2:
                    st.metric("Your Guess", f"{result['user_deg']:.0f}°",
                              delta=f"{result['user_error']:.1f}° error")
                with c3:
                    st.metric("Model Decode", f"{result['model_deg']:.0f}°",
                              delta=f"{result['model_error']:.1f}° error")

                if result['winner'] == "User":
                    st.success("**You win this round!** You beat the model decoder!")
                elif result['winner'] == "Model":
                    st.error("**Model wins!** The decoder was more accurate.")
                else:
                    st.info("**It's a tie!** You matched the model's accuracy.")

                fig_result = plot_polar_comparison(
                    result['true_rad'],
                    result['user_rad'],
                    result['model_rad']
                )
                st.plotly_chart(fig_result, use_container_width=True)

        else:
            st.info("Click **New Round** to start playing!")

    # Game history table
    if st.session_state.game_rounds:
        st.markdown("---")
        st.subheader("Round History")
        fig_table = create_scoreboard_table(st.session_state.game_rounds)
        st.plotly_chart(fig_table, use_container_width=True)


# =========================================================================
# Challenge Modes
# =========================================================================

with tab_challenge:
    st.header("Challenge Modes")
    st.markdown("""
    Test your neural decoding skills in competitive challenge modes!
    Earn achievements and climb the leaderboard.
    """)

    challenge_mgr = st.session_state.challenge_manager
    achievement_mgr = st.session_state.achievement_manager

    if not st.session_state.challenge_active:
        # -- Mode selection --
        st.subheader("Select Challenge Mode")

        col1, col2 = st.columns(2)

        with col1:
            mode_options = {
                "Speed Trial": ChallengeMode.SPEED_TRIAL,
                "Precision": ChallengeMode.PRECISION,
                "Noise Gauntlet": ChallengeMode.NOISE_GAUNTLET,
                "Streak Challenge": ChallengeMode.STREAK
            }

            selected_mode_name = st.radio(
                "Choose your challenge:",
                list(mode_options.keys())
            )
            selected_mode = mode_options[selected_mode_name]

            config = CHALLENGE_CONFIGS[selected_mode]

            st.markdown(f"**{config.name}**")
            st.markdown(config.description)
            st.markdown(f"*Difficulty: {config.difficulty.upper()}*")

            # Show scoring formula for the selected mode
            formula = SCORING_DESCRIPTIONS.get(selected_mode.value, "")
            if formula:
                st.info(formula)

            player_name = st.text_input("Your Name", value="Player", max_chars=30)

            if st.button("Start Challenge!", type="primary"):
                challenge_mgr.start_challenge(selected_mode)
                st.session_state.challenge_active = True
                theta = np.random.uniform(0, 2 * np.pi)
                spikes = simulate_trial(
                    theta, neurons,
                    duration_ms=st.session_state.duration_ms,
                    variance_scale=st.session_state.variance_scale * challenge_mgr.current_noise_level
                )
                st.session_state.challenge_theta = theta
                st.session_state.challenge_spikes = spikes
                st.session_state.player_name = player_name
                st.rerun()

        with col2:
            st.subheader("Leaderboard")
            leaderboard = challenge_mgr.get_leaderboard(selected_mode)
            fig_lb = plot_leaderboard(leaderboard)
            st.plotly_chart(fig_lb, use_container_width=True)

            st.subheader("Your Achievements")
            earned, total = achievement_mgr.get_earned_count()
            st.progress(earned / total if total > 0 else 0)
            st.caption(f"{earned}/{total} achievements earned")

    else:
        # -- Active challenge --
        state = challenge_mgr.get_state()

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader(f"{state['mode'].replace('_', ' ').title()}")

            st.metric("Trials", state['trials_completed'])
            st.metric("Mean Error", f"{state['mean_error']:.1f}°")

            if state['time_remaining'] is not None:
                st.metric("Time Left", f"{state['time_remaining']:.0f}s")

            if state['trials_remaining'] is not None:
                st.metric("Trials Left", state['trials_remaining'])

            if state['mode'] == 'streak':
                st.metric("Current Streak", state['current_streak'])
                st.metric("Best Streak", state['best_streak'])

            if state['mode'] == 'noise_gauntlet':
                st.metric("Noise Level", f"{state['current_noise_level']:.1f}x")

            st.markdown("---")

            if st.button("End Challenge", use_container_width=True):
                result = challenge_mgr.finish_challenge(st.session_state.player_name)
                newly_earned = achievement_mgr.check_achievements(result)

                st.session_state.challenge_active = False
                st.session_state.last_challenge_result = result
                st.session_state.new_achievements = newly_earned
                st.rerun()

        with col1:
            if challenge_mgr.is_challenge_over():
                result = challenge_mgr.finish_challenge(st.session_state.player_name)
                newly_earned = achievement_mgr.check_achievements(result)

                st.success(f"Challenge Complete! Score: **{result.score:.1f}**")

                # Score breakdown
                breakdown = score_breakdown(result)
                cols = st.columns(len(breakdown))
                for col, (label, value) in zip(cols, breakdown.items()):
                    with col:
                        st.metric(label, f"{value:+.1f}" if label != "Total" else f"{value:.1f}")

                st.metric("Trials Completed", result.trials_completed)

                if newly_earned:
                    st.balloons()
                    for achievement in newly_earned:
                        st.success(f"Achievement Unlocked: **{achievement.name}**!")

                st.session_state.challenge_active = False
                st.rerun()
            else:
                st.subheader("Decode This Pattern!")

                fig_pattern = plot_population_bar(
                    st.session_state.challenge_spikes,
                    neurons,
                    true_theta=None,
                    as_polar=True
                )
                st.plotly_chart(fig_pattern, use_container_width=True)

                user_guess = st.number_input(
                    "Your guess (degrees)",
                    min_value=0, max_value=359, value=180, step=1,
                    help="Enter the direction you think this activity represents (0-359)",
                    key="challenge_guess_input"
                )

                if st.button("Submit", type="primary"):
                    true_theta = st.session_state.challenge_theta
                    error = angular_error_degrees(user_guess, radians_to_degrees(true_theta))

                    challenge_mgr.record_trial(error)

                    if error < 20:
                        st.success(f"Great! Error: {error:.1f}°")
                    elif error < 40:
                        st.info(f"Good! Error: {error:.1f}°")
                    else:
                        st.warning(f"Error: {error:.1f}°")

                    theta = np.random.uniform(0, 2 * np.pi)
                    spikes = simulate_trial(
                        theta, neurons,
                        duration_ms=st.session_state.duration_ms,
                        variance_scale=st.session_state.variance_scale * challenge_mgr.current_noise_level
                    )
                    st.session_state.challenge_theta = theta
                    st.session_state.challenge_spikes = spikes
                    st.rerun()
