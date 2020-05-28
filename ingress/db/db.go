package db

import (
	"database/sql"

	log "github.com/sirupsen/logrus"

	// mysql driver
	_ "github.com/go-sql-driver/mysql"
)

//constant
const insertQuery string = "INSERT INTO generals VALUES ()"

// const insertCO2Query string = "INSERT INTO co2 VALUES ()"
var db *sql.DB
var connected bool

type Users struct {
	Name   string
	Score  int64
	images string
}

// connectDB allows to connect to database
func connectDB() *sql.DB {
	cnnstr := "root@tcp(localhost)/general"
	log.Info("try to connect db with %s", cnnstr)
	db, _ := sql.Open("mysql", cnnstr)
	return db
}

func createTable(tableName string) error {
	return nil
}

func getUser(name string) error {
	return nil
}
