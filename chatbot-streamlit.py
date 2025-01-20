import streamlit as st
from chatbot_load import predict_class, get_response, intents
import nltk
import base64
from streamlit_chat import message

# Descargar recursos necesarios
nltk.download('punkt_tab')
nltk.download('wordnet')

st.markdown(
    """
    <style>
        body {
            background-color: black; /* Fondo rojo */
            color: white; /* Color del texto */
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .image-container {
            background-color: #b80711;
            text-align: center; /* Centra la imagen horizontalmente */
        }
    </style>
    
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="image-container"><img src="https://github.com/javyleonhart/Chatbot/raw/9caaf9b9bcb466cc00a3018eb8c9943d18e5263e/activamente.jpeg" alt="imagen"></div>', unsafe_allow_html=True)

st.title("Asistente Virtual")

# Inicializar las variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True
if "user_interacted" not in st.session_state:
    st.session_state.user_interacted = False
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# Mostrar los mensajes guardados en el estado
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖"):
        st.markdown(message["content"])

# Mostrar el primer mensaje del asistente
if st.session_state.first_message:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Hola, soy tu asistente virtual. ¿En qué puedo ayudarte hoy?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, soy tu asistente virtual de Activamente. ¿En qué puedo ayudarte hoy?"})
    st.session_state.first_message = False

# Mostrar sugerencias solo en el primer mensaje y si el usuario no ha interactuado
if st.session_state.show_suggestions:
    st.write("**Sugerencias:**")
    suggestions = [
        "¿En que consiste el programa?",
        "¿Cual es el objetivo?",
        "¿Cuales son los horarios y dias disponibles?",
        "¿Que tipo de entrenamiento se realiza?",
        "¿Cuanto tiempo hace falta entrenar para ver resultados?"
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=suggestion):
            prompt = suggestion
            with st.chat_message("user", avatar='🗣️'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Implementación del algoritmo de IA
            insts = predict_class(prompt)
            res = get_response(insts, intents)
            
            # Respuesta del asistente
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})
            st.session_state.user_interacted = True
            #st.session_state.show_suggestions = False  # Ocultar sugerencias después de seleccionar una
            #break  # Salir del bucle después de seleccionar una sugerencia

# Procesar el mensaje del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    with st.chat_message("user", avatar='🗣️'):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_interacted = True
    #st.session_state.show_suggestions = False  # Ocultar sugerencias después de la interacción del usuario
    
    # Implementación del algoritmo de IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)

    # Respuesta del asistente
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})

# Mostrar el horario de atención y métodos de contacto solo si el usuario no ha interactuado
if not st.session_state.user_interacted:
    with st.expander("Horario de Atención y Contacto"):
        st.markdown("""
        **Horario de Atención:**
        - Lunes a Viernes: 9:00 AM - 6:00 PM
        - Sábado: 10:00 AM - 1:00 PM
        - Domingo: Cerrado

        **Métodos de Contacto:**
        - Teléfono: +54 (264) 123-4567
        - Correo Electrónico: rodri@activamente.com
        - Redes Sociales: [Facebook](https://facebook.com), [Instagram](https://instagram.com), [Twitter](https://twitter.com)
        """)
