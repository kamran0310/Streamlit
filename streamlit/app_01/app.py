import streamlit as st 

#adding title
st.title('My first app heheheh, hurrah')
st.write("simpe text addition")

number = st.slider("pick a number: ", 0, 10)

st.write("You choose: " + str(number))

if st.button('Say Hello'):
    st.write('HI, hello there')
else:
    st.write('Goodbye')
    
 