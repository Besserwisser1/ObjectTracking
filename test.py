import click


@click.command()
@click.argument('input_name', type=click.Path(exists=True))
@click.argument('output_name', type=click.Path())
def main(input_name, output_name):
    print(input_name, output_name)

if __name__=="__main__":
    main()