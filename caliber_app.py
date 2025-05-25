import streamlit as st
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from PIL import Image
import os
from utils_orig import get_openai_api_key
from fpdf import FPDF
import streamlit as st


def sanitize_text(text):
    return (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("…", "...")
            .replace("•", "-")
    )


openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
# llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0,
    openai_api_key=st.secrets["openai_api_key"]
)


st.set_page_config(page_title="CALIBER Leadership Inventory©", layout="centered")
st.title("CALIBER Leadership Inventory©")

st.markdown(
    "<div style='text-align: center; font-size: 0.8em; color: gray; margin-top: -1rem;'>"
    "© 2025 M.A. Lakhani. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)


# Define the 25 items with associated leadership and Hofstede dimensions
item_metadata = [
    (1, 'Reinforcement', 'Uncertainty Avoidance', "I ensure my team clearly understands what success looks like and how it's measured."),
    (2, 'Vision', 'Uncertainty Avoidance', "I see ambiguity as an opportunity to explore new ways forward."),
    (3, 'Communication', 'Individualism', "I check regularly for alignment by encouraging open dialogue and honest dissent."),
    (4, 'Empowerment', 'Power Distance', "I trust my team to make decisions without needing my constant approval."),
    (5, 'Stewardship', 'Long-Term Orientation', "I steward resources with long-term sustainability in mind, not just short-term efficiency."),
    (6, 'Confidence', 'Masculinity', "I project calm and confidence even when situations are volatile."),
    (7, 'Creativity', 'Individualism', "I seek out unconventional ideas actively, even if they challenge the status quo."),
    (8, 'Authenticity', 'Masculinity', "I lead by sharing both my strengths and my uncertainties transparently."),
    (9, 'Competence', 'Long-Term Orientation', "I invest in building my expertise and technical fluency continuously."),
    (10, 'Culture', 'Individualism', "I cultivate a team culture where diverse perspectives are invited and heard."),
    (11, 'Empowerment', 'Power Distance', "I believe hierarchy should be earned through trust and competence, not position."),
    (12, 'Vision', 'Uncertainty Avoidance', "I articulate a compelling vision that guides decisions even in ambiguity."),
    (13, 'Reinforcement', 'Masculinity', "I reward consistent follow-through on agreed-upon goals."),
    (14, 'Creativity', 'Uncertainty Avoidance', "I encourage my team to challenge traditional methods when appropriate."),
    (15, 'Communication', 'Power Distance', "I communicate differently based on individual needs and cultural norms."),
    (16, 'Stewardship', 'Individualism', "I mentor others with the intention of creating leaders, not followers."),
    (17, 'Confidence', 'Masculinity', "I remain calm and decisive during crises, even without full clarity."),
    (18, 'Authenticity', 'Individualism', "I speak up when my values are at odds with organizational directives."),
    (19, 'Competence', 'Masculinity', "I provide frequent, constructive feedback to develop team competence."),
    (20, 'Culture', 'Long-Term Orientation', "I adapt team rituals and symbols to strengthen our unique culture."),
    (21, 'Vision', 'Long-Term Orientation', "I make decisions that balance tradition with transformative change."),
    (22, 'Reinforcement', 'Power Distance', "I ensure recognition is distributed fairly and not reserved for top performers only."),
    (23, 'Culture', 'Individualism', "I challenge everyone to remain open to culturally different ideas."),
    (24, 'Confidence', 'Power Distance', "I am comfortable leading when formal authority is unclear or shared."),
    (25, 'Authenticity', 'Long-Term Orientation', "I reflect on past actions to inform future strategies and leadership growth.")
]

# Load national culture scores from CSV
culture_df = pd.read_csv("culture_scores.csv")

# Standardize column names
culture_df.rename(columns={
    'UAI': 'Uncertainty Avoidance',
    'IDV': 'Individualism',
    'PDI': 'Power Distance',
    'MAS': 'Masculinity'
}, inplace=True)

# # Replaced national_profiles = {

#     'USA': {'Power Distance': 40, 'Individualism': 91, 'Masculinity': 62, 'Uncertainty Avoidance': 46, 'Long-Term Orientation': 26},
#     'Japan': {'Power Distance': 54, 'Individualism': 46, 'Masculinity': 95, 'Uncertainty Avoidance': 92, 'Long-Term Orientation': 88},
#     'Sweden': {'Power Distance': 31, 'Individualism': 71, 'Masculinity': 5, 'Uncertainty Avoidance': 29, 'Long-Term Orientation': 53},
#     'Germany': {'Power Distance': 35, 'Individualism': 67, 'Masculinity': 66, 'Uncertainty Avoidance': 65, 'Long-Term Orientation': 83},
#     'India': {'Power Distance': 77, 'Individualism': 48, 'Masculinity': 56, 'Uncertainty Avoidance': 40, 'Long-Term Orientation': 51}
# }

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = [None] * len(item_metadata)
if 'page' not in st.session_state:
    st.session_state.page = 0

max_page = 5

# Page content
if st.session_state.page == 0:
    st.subheader("Participant Information")
    st.session_state.name = st.text_input("Your Name")
    st.session_state.email = st.text_input("Your Email Address")  # ✅ ADD THIS LINE
    st.session_state.industry = st.text_input("Industry in which you work")
    st.session_state.job_function = st.text_input("Your job function")
    st.session_state.country_work = st.text_input("Country where you currently work")
    st.session_state.birth_country = st.text_input("Country where you were born")

    st.session_state.survey_for = st.radio("Who are you taking this survey for:", ["Myself", "Someone Else"])
    if st.session_state.survey_for == "Someone Else":
        st.session_state.subject_name = st.text_input("Name of the person you are evaluating")
        relation = st.selectbox("Your relationship to that person:", ["The person is my Manager", "The person is my Direct Report", "The person is my Peer", "Other"])
        if relation == "Other":
            st.session_state.relationship = st.text_input("Please describe your relationship")
        else:
            st.session_state.relationship = relation


    known_countries = culture_df['Country'].str.lower().tolist()
    if st.session_state.country_work.lower() not in known_countries:
        # st.warning("⚠️ Country of Work not found in our database. Results will use closest match available.")
        st.session_state.invalid_country_work = True
    else:
        st.session_state.invalid_country_work = False

    if st.session_state.birth_country.lower() not in known_countries:
        # st.warning("⚠️ Country of Birth not found in our database. Results will use closest match available.")
        st.session_state.invalid_birth_country = True
    else:
        st.session_state.invalid_birth_country = False

    occupation_text = st.session_state.get("job_function", "").lower()
    st.session_state.is_retired = any(kw in occupation_text for kw in ["retired", "not working", "unemployed", "none"])

else:
    st.subheader("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree):")
    st.markdown(f"### Page {st.session_state.page} of {max_page}")
    start_idx = (st.session_state.page - 1) * 5
    for i in range(start_idx, start_idx + 5):
        item = item_metadata[i]
        statement = item[3]
        if st.session_state.survey_for == "Someone Else":
            # Simple pronoun replacement
            statement = statement.replace("I am ", "This individual is ")
            statement = statement.replace("I have ", "This individual has ")
            statement = statement.replace("I was ", "This individual was ")
            statement = statement.replace("I will ", "This individual will ")
            statement = statement.replace("I ", "This individual ").replace(" my ", " his/her ")
            if statement.startswith("I"):
                statement = "This individual" + statement[1:]
                # Correct verb conjugation for "This individual" if statement begins with it
            replacements = {
                "ensure": "ensures",
                "steward": "stewards",
                "seek": "seeks",
                "see": "sees",
                "check": "checks",
                "trust": "trusts",
                "steward": "stewards",
                "project": "projects",
                "lead": "leads",
                "invest": "invests",
                "cultivate": "cultivates",
                "believe": "believes",
                "articulate": "articulates",
                "reward": "rewards",
                "encourage": "encourages",
                "communicate": "communicates",
                "mentor": "mentors",
                "remain": "remains",
                "speak": "speaks",
                "provide": "provides",
                "adapt": "adapts",
                "make": "makes",
                "challenge": "challenges",
                "reflect": "reflects",
                "am comfortable leading": "is comfortable leading"
            }
            for k, v in replacements.items():
                prefix = f"This individual {k}"
                if statement.lower().startswith(prefix.lower()):
                    words = statement.split()
                    if len(words) > 2:
                        words[2] = v  # replace the verb directly
                        statement = ' '.join(words)
                    break
                    
            # if k in statement:
            #     statement = statement.replace(k, v, 1)
            # break
        else:
            # For "Myself", use original statement with no change
            statement = item[3]

        st.session_state.responses[i] = st.slider(f"{item[0]}. {statement}", 1, 5, 3, key=f"q{i}")
        # st.session_state.responses[i] = st.slider(f"{item[0]}. {statement}", 1, 5, 3, key=f"q{i}")

# Navigation buttons (bottom only, no form requirement)
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.page > 0:
        if st.button("⬅️ Previous Page"):
            st.session_state.page -= 1
            st.rerun()
with col2:
    if st.session_state.page < max_page:
        if st.button("Next Page ➡️"):
            st.session_state.page += 1
            st.rerun()

from collections import defaultdict

dimension_items = defaultdict(list)
for idx, item in enumerate(item_metadata):
    dimension = item[1]  # Leadership_Dimension
    dimension_items[dimension].append(idx)

# Submit logic
if st.session_state.page == max_page:
    st.subheader("Submit Survey")
    st.write("Please review your answers. When you're ready, click below to submit and download your results.")

    if st.button("Submit Survey"):
        with st.spinner("Creating your report..."):
            df = pd.DataFrame({
                "Question Number": [item[0] for item in item_metadata],
                "Leadership Dimension": [item[1] for item in item_metadata],
                "Hofstede Dimension": [item[2] for item in item_metadata],
                "Statement": [item[3] for item in item_metadata],
                "Response": st.session_state.responses
            })

            # Manual Hofstede dimension calculations (matching Excel logic)
            dimension_custom_scores = {
                'High Uncertainty Avoidance': st.session_state.responses[0] + (5 - st.session_state.responses[1]) + (5 - st.session_state.responses[12]) + (5 - st.session_state.responses[13]),
                'High Individualism': st.session_state.responses[2] + st.session_state.responses[6] + st.session_state.responses[9] + st.session_state.responses[15] + st.session_state.responses[17] + st.session_state.responses[22],
                'High Power Distance': st.session_state.responses[3] + st.session_state.responses[10] + st.session_state.responses[14] + st.session_state.responses[23],
                'Long-Term Orientation': st.session_state.responses[4] + st.session_state.responses[8] + st.session_state.responses[19] + st.session_state.responses[20] + st.session_state.responses[24],
                'High Masculinity': st.session_state.responses[5] + st.session_state.responses[7] + st.session_state.responses[16] + st.session_state.responses[18],
                'High Uncertainty Avoidance PCT': (st.session_state.responses[0] + (5 - st.session_state.responses[1]) + (5 - st.session_state.responses[12]) + (5 - st.session_state.responses[13])-4)/16,
                'High Individualism PCT': (st.session_state.responses[2] + st.session_state.responses[6] + st.session_state.responses[9] + st.session_state.responses[15] + st.session_state.responses[17] + st.session_state.responses[22]-6)/24,
                'High Power Distance PCT': (st.session_state.responses[3] + st.session_state.responses[10] + st.session_state.responses[14] + st.session_state.responses[23]-4)/16,
                'Long-Term Orientation PCT': (st.session_state.responses[4] + st.session_state.responses[8] + st.session_state.responses[19] + st.session_state.responses[20] + st.session_state.responses[24]-5)/20,
                'High Masculinity PCT': (st.session_state.responses[5] + st.session_state.responses[7] + st.session_state.responses[16] + st.session_state.responses[18]-4)/16,
                'Reinforcement': st.session_state.responses[0]+st.session_state.responses[12]+st.session_state.responses[21],
                'Vision': st.session_state.responses[1]+st.session_state.responses[11]+st.session_state.responses[20],
                'Communication':st.session_state.responses[2]+st.session_state.responses[14],
                'Authenticity':st.session_state.responses[7]+st.session_state.responses[17]+st.session_state.responses[24],
                'Competence':st.session_state.responses[8]+st.session_state.responses[18],
                'Confidence':st.session_state.responses[5]+st.session_state.responses[16]+st.session_state.responses[23],
                'Creativity':st.session_state.responses[6]+st.session_state.responses[13],
                'Culture':st.session_state.responses[9]+st.session_state.responses[19]+st.session_state.responses[22],
                'Empowerment':st.session_state.responses[3]+st.session_state.responses[10],
                'Stewardship':st.session_state.responses[4]+st.session_state.responses[15],
                'Reinforcement PCT': (st.session_state.responses[0]+st.session_state.responses[12]+st.session_state.responses[21]-3)/(3*4),
                'Vision PCT': (st.session_state.responses[1]+st.session_state.responses[11]+st.session_state.responses[20]-3)/(3*4),
                'Communication PCT':(st.session_state.responses[2]+st.session_state.responses[14]-2)/(2*4),
                'Authenticity PCT':(st.session_state.responses[7]+st.session_state.responses[17]+st.session_state.responses[24]-3)/(3*4),
                'Competence PCT':(st.session_state.responses[8]+st.session_state.responses[18]-2)/(2*4),
                'Confidence PCT':(st.session_state.responses[5]+st.session_state.responses[16]+st.session_state.responses[23]-3)/(3*4),
                'Creativity PCT':(st.session_state.responses[6]+st.session_state.responses[13]-2)/(2*4),
                'Culture PCT':(st.session_state.responses[9]+st.session_state.responses[19]+st.session_state.responses[22]-3)/(3*4),
                'Empowerment PCT':(st.session_state.responses[3]+st.session_state.responses[10]-2)/(2*4),
                'Stewardship PCT':(st.session_state.responses[4]+st.session_state.responses[15]-2)/(2*4)
            }

            
            # Extract user Hofstede cultural profile
            user_profile = {
                'Uncertainty Avoidance': dimension_custom_scores['High Uncertainty Avoidance PCT'] * 100,
                'Individualism': dimension_custom_scores['High Individualism PCT'] * 100,
                'Power Distance': dimension_custom_scores['High Power Distance PCT'] * 100,
                'Masculinity': dimension_custom_scores['High Masculinity PCT'] * 100
            }

            from scipy.spatial.distance import euclidean

            # Compute Euclidean distance from each country in the dataset
            def compute_distance(row):
                return euclidean([
                    row['Uncertainty Avoidance'],
                    row['Individualism'],
                    row['Power Distance'],
                    row['Masculinity']
                ], list(user_profile.values()))

            culture_df['Distance'] = culture_df.apply(compute_distance, axis=1)
            closest_cultures = culture_df.nsmallest(5, 'Distance')['Country'].tolist()


            leadership_custom_scores = {
                            'Innovation PCT': (dimension_custom_scores['Communication PCT'] + dimension_custom_scores['Vision PCT'] + dimension_custom_scores['Authenticity PCT'] + dimension_custom_scores['Empowerment PCT'] + dimension_custom_scores['Creativity PCT'])/5,
                            'Operations PCT': (dimension_custom_scores['Stewardship PCT'] + dimension_custom_scores['Competence PCT'] + dimension_custom_scores['Confidence PCT'] + dimension_custom_scores['Reinforcement PCT'] + dimension_custom_scores['Culture PCT'])/5,
                            'Overall Leadership PCT': (dimension_custom_scores['Communication PCT'] + dimension_custom_scores['Vision PCT'] + dimension_custom_scores['Authenticity PCT'] + dimension_custom_scores['Empowerment PCT'] + dimension_custom_scores['Creativity PCT'] + dimension_custom_scores['Stewardship PCT'] + dimension_custom_scores['Competence PCT'] + dimension_custom_scores['Confidence PCT'] + dimension_custom_scores['Reinforcement PCT'] + dimension_custom_scores['Culture PCT'])/10
                        }


            # Convert to a DataFrame (transposed to get dimensions as rows)
            dimension_df = pd.DataFrame(list(dimension_custom_scores.items()), columns=['Dimension', 'Score'])

            # Convert to a DataFrame (transposed to get dimensions as rows)
            leadership_df = pd.DataFrame(list(leadership_custom_scores.items()), columns=['Dimension', 'Score'])
            
            # Optional: add a blank row for separation
            blank_row = pd.DataFrame([['', '']], columns=['Dimension', 'Score'])

            

            # Prepare metadata (demographics)
            meta_info = pd.DataFrame({
                'Field': [
                    'Name',
                    'Email',
                    'Job Function',
                    'Industry',
                    'Country of Work',
                    'Country of Birth',
                    'Survey Taken For',
                    'Subject Name',
                    'Relationship'
                ],
                'Value': [
                    st.session_state.get("name", ""),
                    st.session_state.get("email", ""),
                    st.session_state.get("job_function", ""),
                    st.session_state.get("industry", ""),
                    st.session_state.get("country_work", ""),
                    st.session_state.get("birth_country", ""),
                    st.session_state.get("survey_for", ""),
                    st.session_state.get("subject_name", "") if st.session_state.get("survey_for") == "Someone Else" else "",
                    st.session_state.get("relationship", "") if st.session_state.get("survey_for") == "Someone Else" else ""
                ]
            })

            # Optional spacing row
            blank_row = pd.DataFrame([['', '']], columns=['Field', 'Value'])

            # Combine metadata + survey results
            meta_and_scores = pd.concat([meta_info, blank_row], ignore_index=True)

            # Combine original df with new section
            df_combined = pd.concat([df, blank_row, dimension_df, blank_row, leadership_df, blank_row, meta_and_scores], ignore_index=True)


            # Clean name for filename
            participant_name = st.session_state.get("name", "anonymous")
            clean_name = re.sub(r'\W+', '_', participant_name.strip())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"caliber_survey_{clean_name}_{timestamp}.csv"

            df_combined.to_csv(filename, index=False)


            # If survey was for someone else, skip the rest
            if st.session_state.get("survey_for") == "Someone Else":
                st.success("✅ Thank you for providing your assessment. The results have been saved.")
                st.markdown("You may now close this window or return to the home page.")
                st.stop()  # Stop further execution (no report generation)


            # Create and save leadership score plot
            def create_leadership_plot(score, save_path):
                fig, ax = plt.subplots(figsize=(10, 2))

                # Background color zones
                ax.axhspan(0, 1, xmin=0.0, xmax=0.3333, facecolor='#ff9999', alpha=0.5, label='Aspiring Leader')
                ax.axhspan(0, 1, xmin=0.3333, xmax=0.6666, facecolor='#ffe066', alpha=0.5, label='Developing Leader')
                ax.axhspan(0, 1, xmin=0.6666, xmax=1.0, facecolor='#99ff99', alpha=0.5, label='Performing Leader')

                # Score line
                ax.axvline(score, color='black', linewidth=3, label='Your Score')

                # Region Labels
                ax.text(10, 0.8, 'Aspiring Leader', fontsize=10, color='black')
                ax.text(40, 0.8, 'Developing Leader', fontsize=10, color='black')
                ax.text(75, 0.8, 'Performing Leader', fontsize=10, color='black')

                ax.set_title('Overall Leadership Score', fontsize=14, weight='bold')
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xlabel('Score')
                sns.despine(left=True, bottom=True)
                plt.tight_layout()

                fig.savefig(save_path, dpi=150)
                plt.close(fig)

            # Save and show plot
            plot_filename = f"leadership_score_{clean_name}_{timestamp}.png"
            create_leadership_plot(leadership_custom_scores['Overall Leadership PCT']*100, plot_filename)

            # st.image(plot_filename, caption="Your Overall Leadership Score", use_column_width=True)

            # with open(plot_filename, "rb") as f:
            #     st.download_button(
            #         label="📊 Download Your Score Plot",
            #         data=f,
            #         file_name=plot_filename,
            #         mime="image/png"
            #     )

            import streamlit as st
            from PIL import Image

            # Determine leadership level from score
            score_pct = leadership_custom_scores['Overall Leadership PCT'] * 100
            if score_pct <= 33.33:
                level = "Aspiring Leader"
            elif score_pct <= 66.66:
                level = "Developing Leader"
            else:
                level = "Performing Leader"

            # Collect contextual inputs
            participant_role = st.session_state.get("job_function", "a professional")
            participant_industry = st.session_state.get("industry", "their industry")
            country_work = st.session_state.get("country_work", "their country of work")
            birth_country = st.session_state.get("birth_country", "their country of origin")

            # Define the expert agent

            # Define the interpretation task
            summary_description = ("""
            Please note: This is the first page of the CALIBER Leadership Inventory report. In addition to this expert analysis, the full report includes detailed scores, a national culture profile, and specific actions and development recommendations. Encourage the participant to carefully review the complete document.

            """ + 
                f"Write a 1-page report for {participant_name} who works in {participant_industry} as {participant_role}. "
                f"They scored {score_pct:.1f}/100 on the CALIBER Leadership Inventory. "
                f"Label their leadership category as '{level}'. Reflect on the implications of this level of leadership capability "
                f"on team performance and organizational culture within the context of {participant_industry}. Use positive, constructive tone. "
                f"Also take into account that the participant currently works in {country_work} but was born in {birth_country}. "
                f"Comment on how cultural dimensions might influence their leadership style and how cultural awareness can enhance their effectiveness. "
                "Explain why leadership development is vital in their role and industry, and include a motivational call to action for growth." + f" Their leadership practices culturally align best with: {', '.join(closest_cultures)}."
            )

            # Run the crew

            
            summary_prompt = f"""
            Write a 1-page report for {participant_name} who works in {participant_industry} as {participant_role}.
            They scored {score_pct:.1f}/100 on the CALIBER Leadership Inventory.
            Label their leadership category as '{level}'.
            Reflect on implications for team performance and culture within the context of {participant_industry}.
            Include how being born in {birth_country} and working in {country_work} affects leadership style.
            Align analysis with Hofstede cultural dimensions: {', '.join(closest_cultures)}.
            Use the official CALIBER tone: positive, structured, and actionable.
            """
            result = llm.predict(summary_prompt)


            # Show and offer download for report
            # st.subheader("📝 Leadership Expert Report")

            # Insert the images

            # Define image paths
            plot_path = f"leadership_score_{clean_name}_{timestamp}.png"
            bar_chart_path = f"leadership_dimensions_{clean_name}_{timestamp}.png"
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os

            # === Generate and Save Bar Chart ===
            # === Create and Save Hofstede Chart First ===
            hofstede_keys = [
                "High Uncertainty Avoidance PCT",
                "High Individualism PCT",
                "High Power Distance PCT",
                "Long-Term Orientation PCT",
                "High Masculinity PCT"
            ]

            hofstede_scores = [dimension_custom_scores[k] * 100 for k in hofstede_keys]
            hofstede_labels = [
                "Uncertainty Avoidance",
                "Individualism",
                "Power Distance",
                "Long-Term Orientation",
                "Masculinity"
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=hofstede_scores, y=hofstede_labels, palette="Blues_d", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_title("Cultural Dimensions Profile (Hofstede)")
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            sns.despine()

            hofstede_path = f"hofstede_chart_{clean_name}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(hofstede_path, dpi=150)
            plt.close(fig)

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Bar chart for leadership dimensions (Innovation vs Operations)
            dimensions = [
                "Communication PCT", "Vision PCT", "Authenticity PCT", "Empowerment PCT", "Creativity PCT",
                "Stewardship PCT", "Competence PCT", "Confidence PCT", "Reinforcement PCT", "Culture PCT"
            ]

            scores = [
                dimension_custom_scores[dim] * 100 for dim in dimensions
            ]

            labels = [
                "Communication", "Vision", "Authenticity", "Empowerment", "Creativity",
                "Stewardship", "Competence", "Confidence", "Reinforcement", "Culture"
            ]

            category = ["Innovation"] * 5 + ["Operations"] * 5
            palette = sns.color_palette("Set2", 2)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=scores, y=labels, hue=category, dodge=False, palette=palette, ax=ax)
            ax.set_title("Leadership Dimension Scores")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            sns.despine()

            bar_chart_path = f"leadership_dimensions_{clean_name}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(bar_chart_path, dpi=150)
            plt.close(fig)

            # === Next Page Preparation ===
            # Remove inline chart display, only save chart
            # try:
            #     image = Image.open(plot_path)
            #     st.image(image, caption="Overall Leadership Score", use_column_width=True)
            # except Exception as e:
            #     st.warning(f"Could not load image: {e}")
            # Generate PDF with summary and images
            from fpdf import FPDF


            def sanitize_text(text):
                return (
                    text.replace("–", "-")
                        .replace("—", "-")
                        .replace("’", "'")
                        .replace("“", '"')
                        .replace("”", '"')
                        .replace("…", "...")
                        .replace("•", "-")
                )


            pdf_filename = f"leadership_summary_{clean_name}_{timestamp}.pdf"
            class PDFReport(FPDF):
                def footer(self):
                    self.set_y(-10)
                    self.set_font("Arial", "I", 8)
                    self.set_text_color(128)
                    self.cell(0, 10, "© 2025 M.A. Lakhani. All rights reserved.", 0, 0, "C")
                def header(self):
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "CALIBER Leadership Inventory Summary", ln=True, align="C")
                    self.ln(5)

                def chapter_title(self, title):
                    self.set_font("Arial", "B", 11)
                    self.cell(0, 10, title, ln=True, align="L")
                    self.ln(4)

                def chapter_body(self, body):
                    self.set_font("Arial", "", 10)
                    self.multi_cell(0, 5, body)
                    self.ln()

                def add_image(self, path, caption):
                    self.image(path, w=180)
                    self.set_font("Arial", "I", 9)
                    self.cell(0, 5, caption, ln=True, align="C")
                    self.ln(5)

            pdf = PDFReport()
            pdf.add_page()
            pdf.chapter_body(sanitize_text(result))
            pdf.add_image(plot_path, "Overall Leadership Score")
            # Page 2 – Leadership Dimension Breakdown
            pdf.add_page()
            pdf.chapter_title("Leadership Dimension Breakdown")
            pdf.add_image(bar_chart_path, "Dimension Breakdown (Innovation vs Operations)")

            # Define second expert agent for interpretation
            # from crewai import Agent, Task

            # Compose interpretation task for dimensions

            
            page2_prompt = f"""
            Write a summary interpreting the leadership scores in 10 dimensions.
            Separate discussion into Innovation (Communication, Vision, Authenticity, Empowerment, Creativity) and Operations (Stewardship, Competence, Confidence, Reinforcement, Culture).
            Explain the significance of each score, leadership potential, and team/organizational impact.
            Use CALIBER tone, structure, and style.
            """
            page2_result = llm.predict(page2_prompt)

            pdf.chapter_title("Interpretation of Innovation & Operations Dimensions")
            pdf.chapter_body(sanitize_text(page2_result))
            # pdf.add_image(bar_chart_path, "Leadership Dimension Breakdown")
            # Page 3 – National Culture Analysis
            pdf.add_page()
            pdf.chapter_title("Cultural Context and Implications")
            
            culture_prompt = f"""
            Provide a concise analysis of how being born in {birth_country} but currently working in {country_work} might shape leadership expectations.
            Reference Hofstede’s dimensions.
            Include potential cultural tensions or synergies and leadership guidance.
            Use the official CALIBER tone and structure.
            """
            culture_result = llm.predict(culture_prompt)

            pdf.chapter_body(sanitize_text(culture_result))
            pdf.add_image(hofstede_path, "Cultural Dimensions Profile (Hofstede)")

            # Page 4 – Actionable Recommendations
            pdf.add_page()
            pdf.chapter_title("Actionable Development Recommendations")
            
            coach_prompt = f"""
            Write a structured and accessible development plan for {participant_name}.
            Suggest 3–5 growth areas across Innovation and Operations dimensions.
            Provide short rationale for each.
            Comment on Hofstede cultural scores and alignments: {', '.join(closest_cultures)}.
            Offer guidance for cross-cultural adaptability and leadership effectiveness.
            Use CALIBER tone and format.
            """
            coach_result = llm.predict(coach_prompt)

            pdf.chapter_body(sanitize_text(coach_result))

            # Page 5 – Invitation to 360-Degree CALIBER Assessment
            pdf.add_page()
            pdf.chapter_title("Invitation to 360-Degree CALIBER Assessment")
            
            invite_prompt = """
            Write a 1-page summary introducing the CALIBER 360-degree leadership inventory.
            Explain how it exposes biases, highlights cultural fit, tracks progress, and improves self-awareness.
            Encourage multi-source feedback and close with an invitation to contact admin@caliberleadership.com.
            Use CALIBER style.
            """
            invite_result = llm.predict(invite_prompt)

            pdf.chapter_body(sanitize_text(invite_result))

            pdf.output(pdf_filename)

            # Display in Streamlit
            with open(pdf_filename, "rb") as f:
                st.download_button(
                    label="📄 Download Full Leadership Report (PDF)",
                    data=f,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )

            # Show text result
            
            # st.markdown("**Note:** This is only the first page of your CALIBER Leadership Inventory report. It includes an overview of your leadership category, national cultural context, and key development themes. Be sure to review the full report for in-depth scores, your national culture profile, and specific actions and recommendations tailored to you.")

            # st.write(result)

#             # Display top 5 aligned countries
#             st.subheader("🌍 Best Cultural Matches")
#             st.markdown("Your leadership style aligns most closely with these countries:")
#             for country in closest_cultures:
#                 st.markdown(f"- {country}")
#  It includes an overview of your leadership category, national cultural context, and key development themes. Be sure to review the full report for in-depth scores, your national culture profile, and specific actions and recommendations tailored to you.")

#             st.write(result)

            # st.write(result)

            # Display top 5 aligned countries
            # st.subheader("🌍 Best Cultural Matches")
            # st.markdown("Your leadership style aligns most closely with these countries:")
            # for country in closest_cultures:
            #     st.markdown(f"- {country}")


            # # Display bar chart for dimension breakdown
            # try:
            #     bar_image = Image.open(bar_chart_path)
            #     # st.image(bar_image, caption="Dimension Breakdown (Innovation vs Operations)", use_column_width=True)
            # except Exception as e:
            #     # st.warning(f"Could not load bar chart image: {e}")

            # === Display Hofstede Cultural Dimension Chart ===
            # Create and display chart from dimension_custom_scores (Hofstede)
            hofstede_keys = [
                "High Uncertainty Avoidance PCT",
                "High Individualism PCT",
                "High Power Distance PCT",
                "Long-Term Orientation PCT",
                "High Masculinity PCT"
            ]

            hofstede_scores = [dimension_custom_scores[k] * 100 for k in hofstede_keys]
            hofstede_labels = [
                "Uncertainty Avoidance",
                "Individualism",
                "Power Distance",
                "Long-Term Orientation",
                "Masculinity"
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=hofstede_scores, y=hofstede_labels, palette="Blues_d", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_title("Cultural Dimensions Profile (Hofstede)")
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            sns.despine()

            hofstede_path = f"hofstede_chart_{clean_name}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(hofstede_path, dpi=150)
            plt.close(fig)

            # try:
            #     # hofstede_img = Image.open(hofstede_path)
            #     # st.image(hofstede_img, caption="Cultural Dimensions Profile (Hofstede)", use_column_width=True)
            # except Exception as e:
            #     # st.warning(f"Could not load Hofstede chart image: {e}")

            # Display Crew-generated content from page 2–4
            # st.subheader("📊 Interpretation of Innovation & Operations Dimensions")
            # st.write(page2_result)

            # st.subheader("🌍 Cultural Context and Implications")
            # # st.write(culture_result)

            # st.subheader("🎯 Actionable Development Recommendations")
            # # st.write(coach_result)

            report_filename = f"leadership_report_{clean_name}_{timestamp}.txt"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(result)

            with open(pdf_filename, "rb") as f:
                st.markdown("<div style='text-align: center; font-size: 0.8em; color: gray; margin-top: 2rem;'>© 2025 M.A. Lakhani. All rights reserved.</div>", unsafe_allow_html=True)

            # with open(pdf_filename, "rb") as f:
            #     st.download_button(
            #         label="📄 Download Leadership Report (PDF)",
            #         data=f,  # ✅ this was missing
            #         file_name=pdf_filename,
            #         mime="application/pdf"
            #     )

