import streamlit as st
import random

st.title(" ⁉️ Guess the Numer ⁉️")

# initalize the random number
if "secret_number" not in st.session_state:
    st.session_state.secret_number = random.randint(1,100)
    st.session_state.guesses = [0]
    st.session_state.feedback = ""

# take input from the user
guess = st.number_input("Enter your guess:", min_value= 1, max_value=100, step=1)

if st.button("Submit Guess"):
    if guess < 1 or guess > 100:
        st.session_state.feedback = "Out of Bounds. Guess should be bewteen 1 to 100"

    else:
        st.session_state.guesses.append(guess)
        num = st.session_state.secret_number
    
        if guess == num:
            st.session_state.feedback = f"Congaratulations, you guesses it right in {len(st.session_state.guesses)-1} tries!"
    
        else:
            if st.session_state.guesses[-2]:
                if abs(num-guess) < abs(num-st.session_state.guesses[-2]):
                    st.session_state.feedback = "🔥Warmer"
                else:
                    st.session_state.feedback = " 🥶Colder"
                
            else:
                # first guess
                if abs(guess-num) <= 10:
                    st.session_state.feedback = "Warm"
                else:
                    st.session_state.feedback = "Cold"
                    
st.write(st.session_state.feedback)

if st.button("Restart Game"):
    st.session_state.secret_number = random.randint(1,100)
    st.session_state.guesses = [0]
    st.session_state.feedback = ""
    st.experimental_rerun()
    