var upload_avatar = function (file) {
    var fd = new FormData();
    fd.append('uploaded', file);
    $.ajax({
        url: '/upload/avatars',
        method: 'post',
        contentType: false,
        processData: false,
        data: fd,
        success: function (r) {
            if (r.success) {
                var url = r.url;
                window.location.href = url;
            } else {
                log('internal server');
            }
        },
        error: function () {
            console.log('上传失败', file.name);
        }
    });
};

var upload_picture = function (file) {
    var fd = new FormData();
    fd.append('uploaded', file);
    $.ajax({
        url: '/upload/picture',
        method: 'post',
        contentType: false,
        processData: false,
        data: fd,
        success: function (r) {
            if (r.success) {
                var url = r.url;
                template = img_thumnail(url);
                $('#id-div-picturearea').append(template);
                setTimeout(function () {
                    $('p.upload-info').remove()
                }, 3000)
            } else {
                log('internal server');
            }
        },
        error: function () {
            console.log('上传失败', file.name);
        }
    });
};
