import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import base64
import requests

TEST_IMG_IDS_PATH = "data/test_img_ids.json"
TASK_JSON_DIR = "data/"
IMAGE_FOLDER = "data/coco_5000_val_images/"
IMAGES_PER_GROUP = 1000
TOTAL_GROUPS = 5


def push_annotations_to_github(new_annotation):
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    token = st.secrets["GITHUB_TOKEN"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    path = st.secrets.get("GITHUB_PATH", "annotations.json")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        sha = data["sha"]
        existing = json.loads(base64.b64decode(data["content"]).decode("utf-8"))
    elif resp.status_code == 404:
        sha = None
        existing = []
    else:
        raise Exception(f"Error fetching file from GitHub: {resp.text}")

    existing.append(new_annotation)
    content_str = json.dumps(existing, indent=2)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"Add annotation {new_annotation['question_id']}",
        "content": content_b64,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload)
    if put_resp.status_code not in [200, 201]:
        raise Exception(f"GitHub save failed: {put_resp.text}")


TASKS = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count',
         'q_action', 'q_color', 'q_type', 'q_subcategory', 'q_causal']


if "group" not in st.session_state:
    st.session_state.group = "Group 1"
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
if "save_success" not in st.session_state:
    st.session_state.save_success = ""
if "input_state" not in st.session_state:
    st.session_state.input_state = {"question": "", "answer": "", "compositionof": ""}


@st.cache_data
def load_img_ids():
    with open(TEST_IMG_IDS_PATH, "r") as f:
        return json.load(f)

img_ids = load_img_ids()
groups = {f"Group {i+1}": img_ids[i*IMAGES_PER_GROUP:(i+1)*IMAGES_PER_GROUP] for i in range(TOTAL_GROUPS)}


@st.cache_data
def load_all_tasks():
    task_data = {}
    for task in TASKS:
        path = os.path.join(TASK_JSON_DIR, f"karpathy_test_{task}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                task_data[task] = json.load(f)
    return task_data

task_entries = load_all_tasks()


def get_all_qa_for_image(image_id):
    results = []
    for task, entries in task_entries.items():
        for item in entries:
            if item["img_id"] == image_id:
                question = item.get("sent", "")
                answers = [ans["answer"] for ans in item.get("answers", [])]
                question_id = item.get("question_id", "")
                results.append({
                    "task": task,
                    "question_id": question_id,
                    "question": question,
                    "answers": answers
                })
    return results


left_col, right_col = st.columns([2, 3])

with left_col:
    st.markdown("### All Questions/Answers for This Image")
    current_img_id = groups[st.session_state.group][st.session_state.img_index]
    all_qa = get_all_qa_for_image(current_img_id)

    if all_qa:
        for qa in all_qa:
            st.markdown(f"**Task:** `{qa['task']}`")
            st.markdown(f"- **Question ID:** `{qa['question_id']}`")
            st.markdown(f"- **Q:** {qa['question']}")
            st.markdown(f"- **A:** {', '.join(qa['answers'])}")
            st.markdown("---")
    else:
        st.info("No existing Q/A found for this image in loaded tasks.")

with right_col:
    st.markdown("### Annotation Interface")

    group = st.selectbox("Select Group", list(groups.keys()), index=list(groups.keys()).index(st.session_state.group), key="group_select")
    if group != st.session_state.group:
        st.session_state.group = group
        st.session_state.img_index = 0
        st.session_state.save_success = ""
        st.session_state.input_state = {"question": "", "answer": "", "compositionof": ""}
        st.rerun()

    image_ids = groups[st.session_state.group]
    current_img_id = image_ids[st.session_state.img_index]

    image_path = os.path.join(IMAGE_FOLDER, f"{current_img_id}.jpg")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        fixed_width = 300
        w_percent = fixed_width / float(image.size[0])
        h_size = int(float(image.size[1]) * w_percent)
        image = image.resize((fixed_width, h_size), Image.LANCZOS)
        st.image(image, caption=f"Image ID: {current_img_id}")
    else:
        st.error(f"Image not found: {image_path}")

    with st.form("annotation_form", clear_on_submit=True):
        st.text_input("Image ID", value=current_img_id, disabled=True)
        question = st.text_input("Question (e.g., How many yellow truck in the parking lot?)", value=st.session_state.input_state["question"], key="question_input")
        answer = st.text_input("Answer (e.g., two)", value=st.session_state.input_state["answer"], key="answer_input")
        compositionof = st.text_input("Composition of (e.g., count_color)", value=st.session_state.input_state["compositionof"], key="compositionof_input")

        st.session_state.input_state["question"] = question
        st.session_state.input_state["answer"] = answer
        st.session_state.input_state["compositionof"] = compositionof

        submitted = st.form_submit_button("💾 Save Annotation")
        if submitted:
            if not question.strip() or not answer.strip() or not compositionof.strip():
                st.error("❌ All fields must be filled before saving.")
            else:
                now = datetime.now()
                date_part = now.strftime("%d%m")
                time_part = now.strftime("%H%M%S")
                group_num = st.session_state.group.split()[-1]
                question_id = f"{group_num}{date_part}{time_part}"

                new_annotation = {
                    "img_id": current_img_id,
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "compositionof": compositionof.strip(),
                    "group": st.session_state.group,
                    "question_id": question_id
                }

                try:
                    push_annotations_to_github(new_annotation)
                    st.session_state.save_success = f"✅ Annotation saved to GitHub! ID: `{question_id}`"
                    st.session_state.input_state = {"question": "", "answer": "", "compositionof": ""}
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ GitHub save failed: {e}")

        if st.session_state.save_success:
            st.success(st.session_state.save_success)

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("⬅️ Back", disabled=(st.session_state.img_index <= 0)):
            st.session_state.img_index -= 1
            st.session_state.save_success = ""
            st.session_state.input_state = {"question": "", "answer": "", "compositionof": ""}
            st.rerun()
    with nav_col2:
        if st.button("➡️ Next", disabled=(st.session_state.img_index >= len(image_ids) - 1)):
            st.session_state.img_index += 1
            st.session_state.save_success = ""
            st.session_state.input_state = {"question": "", "answer": "", "compositionof": ""}
            st.rerun()

    st.caption(f"Image {st.session_state.img_index + 1} of {len(image_ids)} in {st.session_state.group}")
