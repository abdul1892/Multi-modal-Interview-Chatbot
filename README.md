# Multi-modal-Interview-Chatbot

The recruitment and skill-assessment process continues to rely 
heavily on manual technical interviews that are subjective, difficult to 
scale, and inconsistent across interviewers. Conventional online assessment 
platforms attempt to address scalability, but they typically present a fixed 
bank of text questions, evaluate only the written correctness of an answer, 
and ignore the candidate's spoken communication and confidence.
The system begins by parsing the candidate's resume. A macro 
agent decides which skill to probe next using a depth-first, resumeweighted strategy, while a micro agent maintains a per-skill ability 
estimate and selects the difficulty of each question. Each spoken answer 
is captured, transcribed to text using a Whisper speech-to-text engine, and 
then evaluated through two independent channels.
At the end of the session the system generates a comprehensive 
executive report as a downloadable PDF document. The complete platform 
is implemented as a secure, authenticated web application with persistent 
storage, ensuring that each candidate's interview history is private and 
reproducible.

Authentication and Onboarding<img width="1106" height="552" alt="Screenshot 2026-11-03 225216" src="https://github.com/user-attachments/assets/11ea88e5-4624-457f-bce5-512cec7ddca9" />

Resume Upload and Skill Extraction<img width="1113" height="572" alt="Screenshot 2026-11-03 225242" src="https://github.com/user-attachments/assets/70168c26-04ba-4122-8d67-743ba20060dc" />

The Adaptive Voice Interview<img width="1108" height="582" alt="Screenshot 2026-11-03 225303" src="https://github.com/user-attachments/assets/71a0f25f-e3b1-4687-b974-1eaa419a6864" />

Dual-Channel Evaluation Feedback<img width="1130" height="607" alt="Screenshot 2026-11-03 225329" src="https://github.com/user-attachments/assets/dd379d30-ea17-49c8-948c-c0f124621d78" />

Interview Completion and Execution Report<img width="1062" height="591" alt="Screenshot 2026-11-03 225348" src="https://github.com/user-attachments/assets/6406acc8-8743-459d-8547-bb9845a90440" />






