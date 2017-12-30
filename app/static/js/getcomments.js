var comments_response = function(data, $target){
  if(data.success) {
    // var increment = 1
    // comments_page += increment
    log('success', data);
    var comments = data.comments;
    log('comments', comments)
    comments_template(comments, $target);
  }else {
    log('server error');
  }
};

var comments_template = function(comments, $target){
    var t = comments
        for(var i = 0; i < t.length; i++){
          var template = `
                <hr />
                <div class="well well-sm clearfix" data-id="${t[i].id}">
                  ${t[i].user_name} · ${formatted_time(t[i].created_time)}
                  <br>
                  ${t[i].content}
                </div>
              `;
          $(template).hide().appendTo($target).slideDown("slow")
          }
        }
