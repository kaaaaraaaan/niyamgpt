import streamlit as st
from datetime import datetime

def render_sidebar():
    """Render the sidebar with chat sessions"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2ZM20 16H5.17L4 17.17V4H20V16Z" fill="currentColor"/>
                <path d="M7 9H17M7 13H14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Chat Sessions</span>
        </div>
        """, unsafe_allow_html=True)
        
        # New Chat button
        if st.button("+ New Chat"):
            # Create new chat session
            new_chat_id = st.session_state.next_chat_id
            st.session_state.next_chat_id += 1
            st.session_state.chat_sessions.append({
                "id": new_chat_id,
                "title": f"Session #{new_chat_id}",
                "messages": [],
                "created_at": str(datetime.now())
            })
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        # Display chat sessions
        for session in sorted(st.session_state.chat_sessions, key=lambda x: x['created_at'], reverse=True):
            if st.button(session.get('title', f'Session #{session["id"]}'), key=f"chat_{session['id']}"):
                st.session_state.current_chat_id = session['id']
                st.session_state.messages = session.get('messages', [])
                st.session_state.chat_history = [(msg['content'], msg['content']) for msg in session.get('messages', []) if msg['role'] == 'user']
                st.rerun()
