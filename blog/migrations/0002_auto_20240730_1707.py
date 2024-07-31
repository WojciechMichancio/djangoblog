from django.db import migrations
from django.utils.text import slugify

def generate_slugs(apps, schema_editor):
    Post = apps.get_model('blog', 'Post')
    for post in Post.objects.all():
        if not post.slug:
            post.slug = slugify(post.title)
            original_slug = post.slug
            queryset = Post.objects.all()
            counter = 1
            while queryset.filter(slug=post.slug).exists():
                post.slug = f'{original_slug}-{counter}'
                counter += 1
            post.save()

class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(generate_slugs),
    ]