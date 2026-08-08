import flet
from flet import Checkbox, ElevatedButton, Page, TextField, Row, Column, Dropdown, dropdown, GridView
from flet.ref import Ref

def main(page: Page):
    page.title = "A To-Do List to Organize Your Work & Life"
    
    task_name = Ref[TextField]()


    new_task = TextField(ref=task_name, hint_text="Whats needs to be done?", autofocus=True)

    task_type = Dropdown(
        value="ToDo",
        options=[
            dropdown.Option("ToDo"),
            dropdown.Option("Meeting"),
            dropdown.Option("Appointment")
        ],
        width=200
    )


    def add_clicked(e):
        if new_task.value != "":
            val = task_type.value
            label = val + ": " + task_name.current.value
            checkbox_task = Checkbox(label = label)
            lv.controls.append(checkbox_task)
            #new_task.value = None
            page.update()

    add_task = ElevatedButton(text="Add", on_click=add_clicked)

    def disable_clicked(e):
        for elem in lv.controls:
            if elem.value==True:
                elem.disabled = True
        page.update()


    disable_task = ElevatedButton(text="Disable", on_click=disable_clicked)


    def drop_clicked(e):
        drop_list =[]
        for elem in lv.controls:
            if elem.value==True:
                drop_list.append(elem)
            
        for elem in drop_list:
            lv.controls.remove(elem)

        page.update()
        
    drop_task = ElevatedButton(text="Drop", on_click=drop_clicked)


    def clear_clicked(e):
        lv.controls.clear()
        page.update()

    clear_task = ElevatedButton(text="Clear", on_click=clear_clicked)

    lv = GridView(expand=1,
        runs_count=5,
        max_extent=150,
        child_aspect_ratio=1.0,
        spacing=5,
        run_spacing=5)

    row1 = Row([new_task, task_type], alignment="center")
    row2 = Row([add_task, disable_task, drop_task, clear_task], alignment="center")
    col = Column([row1, row2])

    page.add(col, lv)

flet.app(target=main)